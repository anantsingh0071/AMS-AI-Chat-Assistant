"""
Authentication Module

Provides simple Role-Based Access Control (RBAC)
for the local Streamlit POC.

Later this module can be replaced by:
- Microsoft Entra ID (Azure AD)
- OAuth
- JWT
without changing the rest of the application.
"""

from __future__ import annotations

import streamlit as st

from users import USERS


def login(username: str, password: str) -> bool:
    """
    Validate username and password.
    """

    user = USERS.get(username)

    if user is None:
        return False

    if user["password"] != password:
        return False

    st.session_state.logged_in = True
    st.session_state.username = username
    st.session_state.role = user["role"]

    return True


def logout() -> None:
    """
    Logout current user.
    """

    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.role = None


def is_logged_in() -> bool:
    """
    Check whether the user is logged in.
    """

    return st.session_state.get("logged_in", False)


def has_role(*roles: str) -> bool:
    """
    Check whether the current user has one
    of the allowed roles.
    """

    if not is_logged_in():
        return False

    return st.session_state.get("role") in roles