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
DEFAULT_ACCESS_TOKEN_EXPIRY_SECONDS = 1 * 60  # 1 minute
DEFAULT_REFRESH_TOKEN_EXPIRY_SECONDS = 3 * 60  # 3 minutes

CLIENT_ID_TO_CLIENT_INFO: dict[str, OAuthClientInformationFull] = {
    "claude": OAuthClientInformationFull(
        client_name="claudeai",
        client_id="claude",
        client_secret=None,
        redirect_uris=[AnyUrl("https://claude.ai/api/mcp/auth_callback")],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        scope="claudeai",
        token_endpoint_auth_method="none",
        client_id_issued_at=1750166079,
    ),
    "mcp-inspector-debug": OAuthClientInformationFull(
        client_name="MCP Inspector (Debug)",
        client_id="mcp-inspector-debug",
        client_secret=None,
        redirect_uris=[
            AnyUrl("http://localhost:6274/oauth/callback/debug"),
        ],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        scope="claudeai",
        token_endpoint_auth_method="none",
        client_id_issued_at=1750166079,
    ),
    "mcp-inspector": OAuthClientInformationFull(
        client_name="MCP Inspector",
        client_id="mcp-inspector",
        client_secret=None,
        redirect_uris=[
            AnyUrl("http://localhost:6274/oauth/callback"),
        ],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        scope="claudeai",
        token_endpoint_auth_method="none",
        client_id_issued_at=1750166079,
    ),
    # TODO: doesn't work on VS Code yet
    # "vscode": OAuthClientInformationFull(
    #     client_name="Visual Studio Code",
    #     client_id="vscode",
    #     client_secret=None,
    #     redirect_uris=[
    #         AnyUrl("https://insiders.vscode.dev/redirect"),
    #         AnyUrl("https://vscode.dev/redirect"),
    #         AnyUrl("http://localhost/"),
    #         AnyUrl("http://127.0.0.1/"),
    #         AnyUrl("http://localhost:33418/"),
    #         AnyUrl("http://127.0.0.1:33418/"),
    #     ],
    #     grant_types=["authorization_code", "refresh_token", "urn:ietf:params:oauth:grant-type:device_code"],
    #     response_types=["code"],
    #     scope="claudeai",
    #     token_endpoint_auth_method="none",
    #     client_id_issued_at=1750166079,
    # ),
}


class KeycloakClientSettings(BaseSettings):
    """Settings for the MCP server to communicate with Keycloak."""

    model_config = SettingsConfigDict(env_prefix="MCP_KEYCLOAK_")

    realm: str  # MCP_KEYCLOAK_REALM env var
    client_id: str  # MCP_KEYCLOAK_CLIENT_ID env var
    client_secret: str  # MCP_KEYCLOAK_CLIENT_SECRET env var
    client_root_url: AnyHttpUrl  # MCP_KEYCLOAK_CLIENT_ROOT_URL env var

    keycloak_server_url: AnyHttpUrl = AnyHttpUrl(f"{os.environ['MOSTLY_BASE_URL']}/auth")
    keycloak_scope: str = "openid"

    # NOTE: Claude's clients explicitly requests the `claudeai` scope, while other clients don't specify any
    # Therefore we just use this for all clients for now
    mcp_scope: str = "claudeai"

    @property
    def client_redirect_uris(self) -> str:
        return f"{self.client_root_url._url}oauth/callback"

    @property
    def keycloak_auth_url(self) -> str:
        return f"{self.keycloak_server_url}/realms/{self.realm}/protocol/openid-connect/auth"

    @property
    def keycloak_token_url(self) -> str:
        return f"{self.keycloak_server_url}/realms/{self.realm}/protocol/openid-connect/token"


class KeycloakOAuthProvider(OAuthProvider):
    def __init__(self, host: str, port: int):
        logger.info(f"initializing keycloak OAuth provider for {host}:{port}")
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
        
        logger.info("keycloak OAuth provider initialized")

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        logger.debug(f"get_client: {client_id}")
        # fetch if the client is already dynamically registered, otherwise try to fetch the static client info
        return self.clients.get(client_id, CLIENT_ID_TO_CLIENT_INFO.get(client_id))

    async def register_client(self, client_info: OAuthClientInformationFull):
        logger.info(f"register_client: {client_info.client_id}")
        self.clients[client_info.client_id] = client_info

    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        logger.info(f"authorize: client={client.client_id}")
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
            f"{self.settings.keycloak_auth_url}"
            f"?client_id={self.settings.client_id}"
            f"&redirect_uri={self.settings.client_redirect_uris}"
            f"&scope={self.settings.keycloak_scope}"
            f"&response_type=code"
            f"&state={state}"
        )

        return auth_url

    async def handle_callback(self, code: str, state: str) -> str:
        logger.info(f"handle_callback called")
        
        state_data = self.state_mapping.get(state)
        if not state_data:
            logger.error(f"invalid state parameter")
            raise HTTPException(400, "Invalid state parameter")

        redirect_uri = state_data["redirect_uri"]
        code_challenge = state_data["code_challenge"]
        redirect_uri_provided_explicitly = state_data["redirect_uri_provided_explicitly"] == "True"
        client_id = state_data["client_id"]

        logger.info("exchanging code for token with keycloak")
        # Exchange code for token with Keycloak
        async with create_mcp_http_client() as client:
            response = await client.post(
                self.settings.keycloak_token_url,
                data={
                    "grant_type": "authorization_code",
                    "client_id": self.settings.client_id,
                    "client_secret": self.settings.client_secret,
                    "code": code,
                    "redirect_uri": self.settings.client_redirect_uris,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

            if response.status_code != 200:
                error_msg = f"Failed to exchange code for token: {response.text}"
                logger.error(error_msg)
                raise HTTPException(400, error_msg)

            data = response.json()
            if "error" in data:
                error_msg = data.get("error_description", data["error"])
                logger.error(f"keycloak token exchange error: {error_msg}")
                raise HTTPException(400, error_msg)

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
        redirect_url = construct_redirect_uri(redirect_uri, code=new_code, state=state)
        logger.info("handle_callback successful")
        return redirect_url

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        return self.auth_codes.get(authorization_code)

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        if authorization_code.code not in self.auth_codes:
            raise ValueError("Invalid authorization code")

        # Generate MCP access token
        mcp_access_token = f"mcp_{secrets.token_hex(32)}"
        mcp_refresh_token = f"mcp_{secrets.token_hex(32)}"
        logger.info(f"mcp_access_token: {mcp_access_token[:10]}****")
        logger.info(f"mcp_refresh_token: {mcp_refresh_token[:10]}****")

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
        logger.info(f"load_access_token: {token[:10]}****")
        token_obj = self.access_token_objs.get(token)
        if token_obj:
            if token_obj.expires_at is not None and token_obj.expires_at < time.time():
                self._revoke_access_token(token=token_obj.token)  # Clean up expired
                return None
            return token_obj
        return None

    async def load_refresh_token(self, client: OAuthClientInformationFull, refresh_token: str) -> RefreshToken | None:
        logger.info(f"load_refresh_token: {refresh_token[:10]}****")
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
        logger.info(f"new access token: {new_access_token[:10]}****")
        logger.info(f"new refresh token: {new_refresh_token[:10]}****")

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
        logger.info("refreshing keycloak tokens")
        async with create_mcp_http_client() as http_client:
            response = await http_client.post(
                self.settings.keycloak_token_url,
                data={
                    "grant_type": "refresh_token",
                    "client_id": self.settings.client_id,
                    "client_secret": self.settings.client_secret,
                    "refresh_token": keycloak_refresh_token,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

            if response.status_code != 200:
                logger.error(f"keycloak refresh failed: {response.status_code}")
                raise HTTPException(400, f"Failed to exchange code for token: {response.text}")

            data = response.json()
            if "error" in data:
                logger.error(f"keycloak refresh error: {data.get('error_description', data['error'])}")
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
        logger.info(f"revoking access token: {token[:10]}****")
        self.access_token_objs.pop(token, None)
        self._mcp_access_to_keycloak_access_map.pop(token, None)

    def _revoke_refresh_token(self, token: str) -> tuple[str, str]:
        logger.info(f"revoking refresh token: {token[:10]}****")
        self.refresh_token_objs.pop(token, None)
        keycloak_access_token = self._mcp_refresh_to_keycloak_access_map.pop(token, None)
        keycloak_refresh_token = self._mcp_refresh_to_keycloak_refresh_map.pop(token, None)

        # revoke associated access token as well
        associated_access = self._mcp_refresh_to_mcp_access_map.pop(token, None)
        logger.info(f"revoking associated access token: {associated_access[:10]}****")
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
