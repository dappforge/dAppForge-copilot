import os
import boto3
from aws_cdk import (
    Stack,
    aws_iam as iam,
)
from constructs import Construct
from dotenv import load_dotenv

load_dotenv(".env")
load_dotenv("../.env")

PREFIX = os.getenv("PREFIX", "")
DEPLOY_REPO_NAME = os.getenv("DEPLOY_REPO_NAME")
DEPLOY_ROLE_NAME = os.getenv("DEPLOY_ROLE_NAME", f"{PREFIX}GitHubDeploymentIAMRole")
DEPLOY_OPEN_ID_PROVIDER_ARN = os.getenv("DEPLOY_OPEN_ID_PROVIDER_ARN")
iam_client = boto3.client("iam")


class GitHubAuthStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        if DEPLOY_REPO_NAME is None:
            raise ValueError("DEPLOY_REPO_NAME is not set")

        if DEPLOY_OPEN_ID_PROVIDER_ARN:
            # Reading existing provider
            github_provider = (
                iam.OpenIdConnectProvider.from_open_id_connect_provider_arn(
                    self,
                    f"{PREFIX}GitHubProvider",
                    DEPLOY_OPEN_ID_PROVIDER_ARN,
                )
            )
        else:
            # Create provider
            github_provider = iam.OpenIdConnectProvider(
                self,
                f"{PREFIX}GitHubProvider",
                url="https://token.actions.githubusercontent.com",
                client_ids=[
                    "sts.amazonaws.com",
                ],
                thumbprints=[
                    "6938fd4d98bab03faadb97b34396831e3780aea1",
                    "1c58a3a8518e8759bf075b76b750d4f2df264fcd",
                    "ffffffffffffffffffffffffffffffffffffffff",
                ],
            )
        self.github_provider = github_provider

        # Creating role
        github_role = iam.Role(
            self,
            f"{PREFIX}GitHubDeploymentRole",
            role_name=DEPLOY_ROLE_NAME,
            assumed_by=iam.WebIdentityPrincipal(
                github_provider.open_id_connect_provider_arn,
                conditions={
                    "StringEquals": {
                        "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
                    },
                    "StringLike": {
                        "token.actions.githubusercontent.com:sub": f"repo:{DEPLOY_REPO_NAME}:*",
                    },
                },
            ),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AdministratorAccess")
            ],
        )
        self.github_role = github_role
