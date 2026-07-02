from __future__ import annotations

import os
import secrets
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

_security = HTTPBasic(auto_error=False)


def optional_basic_auth(
    credentials: Optional[HTTPBasicCredentials] = Depends(_security),
) -> None:
    user = os.environ.get("WEB_BASIC_AUTH_USER", "")
    password = os.environ.get("WEB_BASIC_AUTH_PASS", "")
    if not user or not password:
        return

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Basic"},
        )

    ok_user = secrets.compare_digest(credentials.username, user)
    ok_pass = secrets.compare_digest(credentials.password, password)
    if not (ok_user and ok_pass):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
