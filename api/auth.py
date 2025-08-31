import os
import bcrypt
import yaml
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()

def load_users_from_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
        users = {user['username']: user['password'].encode('utf-8') for user in data['users']}
        return users

# Load users from YAML file
users_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'users.yaml')
users = load_users_from_yaml(users_file_path)

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    hashed_password = users.get(credentials.username)
    if not hashed_password or not bcrypt.checkpw(credentials.password.encode('utf-8'), hashed_password.encode('utf-8')):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username