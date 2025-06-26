# Copyright 2025 MOSTLY AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import secrets
import time
from urllib.parse import quote

from fastmcp.server.auth.auth import OAuthProvider
from pydantic import AnyHttpUrl, AnyUrl
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.exceptions import HTTPException

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    RefreshToken,
    TokenError,
    construct_redirect_uri,
)

# from mcp.server.auth.handlers.token import
from mcp.server.auth.settings import ClientRegistrationOptions, RevocationOptions
from mcp.shared._httpx_utils import create_mcp_http_client
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

logger = logging.getLogger(__name__)

DEFAULT_AUTH_CODE_EXPIRY_SECONDS = 5 * 60  # 5 minutes
DEFAULT_ACCESS_TOKEN_EXPIRY_SECONDS = 5 * 60  # 5 minute
DEFAULT_REFRESH_TOKEN_EXPIRY_SECONDS = 15 * 60  # 15 minutes

CLIENT_ID_TO_CLIENT_INFO: dict[str, OAuthClientInformationFull] = {
    "claude": OAuthClientInformationFull(
        client_name="claudeai",
        client_id="claude",
        redirect_uris=[AnyUrl("https://claude.ai/api/mcp/auth_callback")],
        scope="claudeai",
        token_endpoint_auth_method="none",
        client_id_issued_at=1750166079,
    ),
    "mcp-inspector-debug": OAuthClientInformationFull(
        client_name="MCP Inspector (Debug)",
        client_id="mcp-inspector-debug",
        redirect_uris=[AnyUrl("http://localhost:6274/oauth/callback/debug")],
        scope="claudeai",
        token_endpoint_auth_method="none",
        client_id_issued_at=1750166079,
    ),
    "mcp-inspector": OAuthClientInformationFull(
        client_name="MCP Inspector",
        client_id="mcp-inspector",
        redirect_uris=[AnyUrl("http://localhost:6274/oauth/callback")],
        scope="claudeai",
        token_endpoint_auth_method="none",
        client_id_issued_at=1750166079,
    ),
    "vscode": OAuthClientInformationFull(
        client_name="Visual Studio Code",
        client_id="vscode",
        redirect_uris=[
            AnyUrl("https://insiders.vscode.dev/redirect"),
            AnyUrl("https://vscode.dev/redirect"),
            AnyUrl("http://localhost/"),
            AnyUrl("http://127.0.0.1/"),
            AnyUrl("http://localhost:33418/"),
            AnyUrl("http://127.0.0.1:33418/"),
        ],
        scope="claudeai",
        token_endpoint_auth_method="none",
        client_id_issued_at=1750166079,
    ),
    # NOTE: currently, users must name the server as `mostlyai` in MCP config, so that it will use this redirect URI
    "cursor": OAuthClientInformationFull(
        client_name="Cursor",
        client_id="cursor",
        redirect_uris=[AnyUrl("cursor://anysphere.cursor-retrieval/oauth/user-mostlyai/callback")],
        scope="claudeai",
        token_endpoint_auth_method="none",
        client_id_issued_at=1750166079,
    ),
}


class KeycloakClientSettings(BaseSettings):
    """Settings for the MCP server to communicate with Keycloak."""

    model_config = SettingsConfigDict(env_prefix="MCP_KEYCLOAK_")

    realm: str  # MCP_KEYCLOAK_REALM env var
    client_id: str  # MCP_KEYCLOAK_CLIENT_ID env var
    client_root_url: AnyHttpUrl  # MCP_KEYCLOAK_CLIENT_ROOT_URL env var

    auth_url: AnyHttpUrl  # MCP_KEYCLOAK_AUTH_URL env var
    keycloak_scope: str = "openid"

    # NOTE: Claude's clients explicitly requests the `claudeai` scope, while other clients don't specify any
    # Therefore we just use this for all clients for now
    mcp_scope: str = "claudeai"

    @property
    def client_redirect_uris(self) -> str:
        return str(self.client_root_url).rstrip("/") + "/oauth/callback"

    @property
    def keycloak_authorization_endpoint(self) -> str:
        return str(self.auth_url).rstrip("/") + f"/realms/{self.realm}/protocol/openid-connect/auth"

    @property
    def keycloak_token_endpoint(self) -> str:
        return str(self.auth_url).rstrip("/") + f"/realms/{self.realm}/protocol/openid-connect/token"


class KeycloakOAuthProvider(OAuthProvider):
    def __init__(self, host: str, port: int):
        # since MCP server itself is a client of Keycloak, the client root URL is the same as the MCP server public URL
        mcp_server_public_url = AnyHttpUrl(os.getenv("MCP_KEYCLOAK_CLIENT_ROOT_URL", f"http://{host}:{port}"))
        self.settings = KeycloakClientSettings(client_root_url=mcp_server_public_url)
        client_registration_options = ClientRegistrationOptions(
            enabled=True,
            valid_scopes=[self.settings.mcp_scope],
            default_scopes=[self.settings.mcp_scope],
        )
        revocation_options = RevocationOptions(enabled=True)
        required_scopes = []
        super().__init__(
            issuer_url=mcp_server_public_url,
            client_registration_options=client_registration_options,
            revocation_options=revocation_options,
            required_scopes=required_scopes,
        )

        self.clients: dict[str, OAuthClientInformationFull] = {}
        self.auth_codes: dict[str, AuthorizationCode] = {}
        self.access_token_objs: dict[str, AccessToken] = {}
        self.refresh_token_objs: dict[str, RefreshToken] = {}
        self.state_mapping: dict[str, dict[str, str]] = {}
        self._mcp_access_to_keycloak_access_map: dict[str, str] = {}
        self._mcp_refresh_to_keycloak_access_map: dict[str, str] = {}
        self._mcp_refresh_to_keycloak_refresh_map: dict[str, str] = {}
        self._auth_code_to_keycloak_access_map: dict[str, str] = {}
        self._auth_code_to_keycloak_refresh_map: dict[str, str] = {}

        # For revoking associated tokens
        self._mcp_refresh_to_mcp_access_map: dict[str, str] = {}

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        logger.info(f"get_client: {client_id}")
        # the registered client info still lives in the server's memory
        if client_id in self.clients:
            return self.clients[client_id]
        # the client info does not live in the server's memory, but we can still recover it as they're static
        if client_id in CLIENT_ID_TO_CLIENT_INFO:
            return CLIENT_ID_TO_CLIENT_INFO[client_id]
        # best effort to recover the client info (assume that the client is redirecting to localhost)
        return OAuthClientInformationFull(
            client_name="Dynamically-Registered MCP Client",
            client_id=client_id,
            client_secret=None,
            redirect_uris=[AnyUrl(f"http://localhost:{port}/oauth/callback") for port in range(1024, 65535)],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            scope="claudeai",
            token_endpoint_auth_method="none",
            client_id_issued_at=1750166079,
        )

    async def register_client(self, client_info: OAuthClientInformationFull):
        logger.info(f"register_client: {client_info}")
        self.clients[client_info.client_id] = client_info

    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        state = params.state or secrets.token_hex(16)
        # Store the state mapping
        self.state_mapping[state] = {
            "redirect_uri": str(params.redirect_uri),
            "code_challenge": params.code_challenge,
            "redirect_uri_provided_explicitly": str(params.redirect_uri_provided_explicitly),
            "client_id": client.client_id,
        }

        # Build Keycloak authorization URL
        auth_url = (
            f"{self.settings.keycloak_authorization_endpoint}"
            f"?client_id={self.settings.client_id}"
            f"&redirect_uri={quote(self.settings.client_redirect_uris, safe='')}"
            f"&scope={self.settings.keycloak_scope}"
            f"&response_type=code"
            f"&state={quote(state, safe='')}"
        )

        return auth_url

    async def handle_callback(self, code: str, state: str) -> str:
        state_data = self.state_mapping.get(state)
        if not state_data:
            raise HTTPException(400, "Invalid state parameter")

        redirect_uri = state_data["redirect_uri"]
        code_challenge = state_data["code_challenge"]
        redirect_uri_provided_explicitly = state_data["redirect_uri_provided_explicitly"] == "True"
        client_id = state_data["client_id"]

        # Exchange code for token with Keycloak
        async with create_mcp_http_client() as client:
            response = await client.post(
                self.settings.keycloak_token_endpoint,
                data={
                    "grant_type": "authorization_code",
                    "client_id": self.settings.client_id,
                    "code": code,
                    "redirect_uri": self.settings.client_redirect_uris,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

            if response.status_code != 200:
                raise HTTPException(400, f"Failed to exchange code for token: {response.text}")

            data = response.json()
            if "error" in data:
                raise HTTPException(400, data.get("error_description", data["error"]))

            keycloak_access_token = data["access_token"]
            keycloak_refresh_token = data["refresh_token"]

            # Create MCP authorization code
            new_code = f"mcp_{secrets.token_hex(16)}"
            self.auth_codes[new_code] = AuthorizationCode(
                code=new_code,
                client_id=client_id,
                redirect_uri=AnyUrl(redirect_uri),
                redirect_uri_provided_explicitly=redirect_uri_provided_explicitly,
                expires_at=time.time() + DEFAULT_AUTH_CODE_EXPIRY_SECONDS,
                scopes=[self.settings.mcp_scope],
                code_challenge=code_challenge,
            )

            # Store Keycloak token - we'll map the MCP token to this later
            self.access_token_objs[keycloak_access_token] = AccessToken(
                token=keycloak_access_token,
                client_id=client_id,
                scopes=self.settings.keycloak_scope.split(),
                expires_at=None,  # decided by Keycloak server
            )

            self.refresh_token_objs[keycloak_refresh_token] = RefreshToken(
                token=keycloak_refresh_token,
                client_id=client_id,
                scopes=self.settings.keycloak_scope.split(),
                expires_at=None,  # decided by Keycloak server
            )

            self._auth_code_to_keycloak_access_map[new_code] = keycloak_access_token
            self._auth_code_to_keycloak_refresh_map[new_code] = keycloak_refresh_token

        self.state_mapping.pop(state, None)
        return construct_redirect_uri(redirect_uri, code=new_code, state=state)

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        return self.auth_codes.get(authorization_code)

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        logger.info("exchange_authorization_code")
        if authorization_code.code not in self.auth_codes:
            raise ValueError("Invalid authorization code")

        # Generate MCP access token
        mcp_access_token = f"mcp_{secrets.token_hex(32)}"
        mcp_refresh_token = f"mcp_{secrets.token_hex(32)}"

        mcp_access_token_expires_at = int(time.time() + DEFAULT_ACCESS_TOKEN_EXPIRY_SECONDS)

        mcp_refresh_token_expires_at = None
        if DEFAULT_REFRESH_TOKEN_EXPIRY_SECONDS is not None:
            mcp_refresh_token_expires_at = int(time.time() + DEFAULT_REFRESH_TOKEN_EXPIRY_SECONDS)

        # Store MCP token
        self.access_token_objs[mcp_access_token] = AccessToken(
            token=mcp_access_token,
            client_id=client.client_id,
            scopes=authorization_code.scopes,
            expires_at=mcp_access_token_expires_at,
        )

        self.refresh_token_objs[mcp_refresh_token] = RefreshToken(
            token=mcp_refresh_token,
            client_id=client.client_id,
            scopes=authorization_code.scopes,
            expires_at=mcp_refresh_token_expires_at,
        )

        self._mcp_refresh_to_mcp_access_map[mcp_refresh_token] = mcp_access_token

        # Find Keycloak token for this client
        keycloak_access_token = self._auth_code_to_keycloak_access_map[authorization_code.code]
        keycloak_refresh_token = self._auth_code_to_keycloak_refresh_map[authorization_code.code]

        # Store mapping between MCP token and Keycloak token
        self._mcp_access_to_keycloak_access_map[mcp_access_token] = keycloak_access_token
        self._mcp_refresh_to_keycloak_access_map[mcp_refresh_token] = keycloak_access_token
        self._mcp_refresh_to_keycloak_refresh_map[mcp_refresh_token] = keycloak_refresh_token

        self.auth_codes.pop(authorization_code.code, None)
        self._auth_code_to_keycloak_access_map.pop(authorization_code.code, None)
        self._auth_code_to_keycloak_refresh_map.pop(authorization_code.code, None)

        return OAuthToken(
            access_token=mcp_access_token,
            token_type="Bearer",
            expires_in=DEFAULT_ACCESS_TOKEN_EXPIRY_SECONDS,
            refresh_token=mcp_refresh_token,
            scope=" ".join(authorization_code.scopes),
        )

    async def load_access_token(self, token: str) -> AccessToken | None:
        token_obj = self.access_token_objs.get(token)
        if token_obj:
            if token_obj.expires_at is not None and token_obj.expires_at < time.time():
                self._revoke_access_token(token=token_obj.token)  # Clean up expired
                return None
            return token_obj
        return None

    async def load_refresh_token(self, client: OAuthClientInformationFull, refresh_token: str) -> RefreshToken | None:
        token_obj = self.refresh_token_objs.get(refresh_token)
        if token_obj:
            if token_obj.client_id != client.client_id:
                return None
            if token_obj.expires_at is not None and token_obj.expires_at < time.time():
                self._revoke_refresh_token(token=token_obj.token)  # Clean up expired
                return None
            return token_obj
        return None

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,  # This is the RefreshToken object, already loaded
        scopes: list[str],  # Requested scopes for the new access token
    ) -> OAuthToken:
        logger.info("exchange_refresh_token")
        # Validate scopes: requested scopes must be a subset of original scopes
        original_scopes = set(refresh_token.scopes)
        requested_scopes = set(scopes)
        if not requested_scopes.issubset(original_scopes):
            raise TokenError(
                "invalid_scope",
                "Requested scopes exceed those authorized by the refresh token.",
            )

        # Invalidate old refresh token and its associated access token (rotation)
        keycloak_access_token, keycloak_refresh_token = self._revoke_refresh_token(token=refresh_token.token)

        # Issue new tokens
        new_access_token = f"mcp_{secrets.token_hex(32)}"
        new_refresh_token = f"mcp_{secrets.token_hex(32)}"

        access_token_expires_at = int(time.time() + DEFAULT_ACCESS_TOKEN_EXPIRY_SECONDS)

        refresh_token_expires_at = refresh_token.expires_at

        self.access_token_objs[new_access_token] = AccessToken(
            token=new_access_token,
            client_id=client.client_id,
            scopes=scopes,  # Use newly requested (and validated) scopes
            expires_at=access_token_expires_at,
        )
        self.refresh_token_objs[new_refresh_token] = RefreshToken(
            token=new_refresh_token,
            client_id=client.client_id,
            scopes=scopes,  # New refresh token also gets these scopes
            expires_at=refresh_token_expires_at,
        )

        # refresh keycloak tokens
        async with create_mcp_http_client() as http_client:
            response = await http_client.post(
                self.settings.keycloak_token_endpoint,
                data={
                    "grant_type": "refresh_token",
                    "client_id": self.settings.client_id,
                    "refresh_token": keycloak_refresh_token,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

            if response.status_code != 200:
                logger.error(f"400: {response.text}")
                raise HTTPException(400, f"Failed to exchange code for token: {response.text}")

            data = response.json()
            if "error" in data:
                logger.error(f"400: {data.get('error_description', data['error'])}")
                raise HTTPException(400, data.get("error_description", data["error"]))

            new_keycloak_access_token = data["access_token"]
            new_keycloak_refresh_token = data["refresh_token"]

        self.access_token_objs[new_keycloak_access_token] = AccessToken(
            token=new_keycloak_access_token,
            client_id=client.client_id,
            scopes=self.settings.keycloak_scope.split(),
            expires_at=None,
        )

        self.refresh_token_objs[new_keycloak_refresh_token] = RefreshToken(
            token=new_keycloak_refresh_token,
            client_id=client.client_id,
            scopes=self.settings.keycloak_scope.split(),
            expires_at=None,
        )
        self.access_token_objs.pop(keycloak_access_token, None)
        self.refresh_token_objs.pop(keycloak_refresh_token, None)

        # link the new tokens to the existing keycloak tokens
        self._mcp_access_to_keycloak_access_map[new_access_token] = new_keycloak_access_token
        self._mcp_refresh_to_keycloak_access_map[new_refresh_token] = new_keycloak_access_token
        self._mcp_refresh_to_keycloak_refresh_map[new_refresh_token] = new_keycloak_refresh_token
        self._mcp_refresh_to_mcp_access_map[new_refresh_token] = new_access_token

        return OAuthToken(
            access_token=new_access_token,
            token_type="Bearer",
            expires_in=DEFAULT_ACCESS_TOKEN_EXPIRY_SECONDS,
            refresh_token=new_refresh_token,
            scope=" ".join(scopes),
        )

    def _revoke_access_token(self, token: str) -> None:
        self.access_token_objs.pop(token, None)
        self._mcp_access_to_keycloak_access_map.pop(token, None)

    def _revoke_refresh_token(self, token: str) -> tuple[str, str]:
        self.refresh_token_objs.pop(token, None)
        keycloak_access_token = self._mcp_refresh_to_keycloak_access_map.pop(token, None)
        keycloak_refresh_token = self._mcp_refresh_to_keycloak_refresh_map.pop(token, None)

        # revoke associated access token as well
        associated_access = self._mcp_refresh_to_mcp_access_map.pop(token, None)
        if associated_access:
            self.access_token_objs.pop(associated_access, None)
            self._mcp_access_to_keycloak_access_map.pop(associated_access, None)

        return keycloak_access_token, keycloak_refresh_token

    async def revoke_token(
        self,
        token: AccessToken | RefreshToken,
    ) -> None:
        if isinstance(token, AccessToken):
            self._revoke_access_token(token=token.token)
        elif isinstance(token, RefreshToken):
            self._revoke_refresh_token(token=token.token)
