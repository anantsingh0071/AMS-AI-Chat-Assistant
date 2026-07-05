"""
Mock User Store

This simulates a user database.

Later this can be replaced by:

- Azure Active Directory (Microsoft Entra ID)
- SQL Database
- Azure SQL
- Microsoft Graph
"""

USERS = {
    "admin": {
        "password": "admin123",
        "role": "admin",
    },

    "marketing": {
        "password": "marketing123",
        "role": "marketing",
    },

    "sales": {
        "password": "sales123",
        "role": "sales",
    },

    "guest": {
        "password": "guest123",
        "role": "guest",
    },
}