import json
import os
from typing import Annotated

import boto3
from botocore.exceptions import ClientError
from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

api_key_param = os.environ.get("API_KEY_PARAM_NAME")
api_key_secret_arn = os.environ.get("API_KEY_SECRET_ARN")
api_key_env = os.environ.get("API_KEY")
if api_key_param:
    # For backward compatibility.
    # Please now use secrets manager instead.
    ssm = boto3.client("ssm")
    api_key = ssm.get_parameter(Name=api_key_param, WithDecryption=True)["Parameter"]["Value"]
elif api_key_secret_arn:
    sm = boto3.client("secretsmanager")
    try:
        response = sm.get_secret_value(SecretId=api_key_secret_arn)
        if "SecretString" in response:
            secret = json.loads(response["SecretString"])
            api_key = secret["api_key"]
    except ClientError:
        raise RuntimeError("Unable to retrieve API KEY, please ensure the secret ARN is correct")
    except KeyError:
        raise RuntimeError('Please ensure the secret contains a "api_key" field')
elif api_key_env:
    api_key = api_key_env
else:
    raise RuntimeError(
        "API Key is not configured. Please set up your API Key."
    )

security = HTTPBearer()


def api_key_auth(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
):
    if credentials.credentials != api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")


# Anthropic clients send the key as x-api-key (ANTHROPIC_API_KEY) or as a bearer token
# (ANTHROPIC_AUTH_TOKEN), so the Messages API accepts either.
x_api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)
optional_bearer = HTTPBearer(auto_error=False)


def anthropic_api_key_auth(
    x_api_key: Annotated[str | None, Depends(x_api_key_header)],
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(optional_bearer)],
):
    key = x_api_key or (credentials.credentials if credentials else None)
    if key != api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")
