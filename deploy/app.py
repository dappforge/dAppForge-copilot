#!/usr/bin/env python3
import os
import aws_cdk as cdk
from dotenv import load_dotenv

load_dotenv('.env')
load_dotenv('../.env')

from deploy.github_stack import GitHubAuthStack
from deploy.resources_stack import ResourcesStack
from deploy.services_stack import ServicesStack

PREFIX = os.getenv('PREFIX', '')
env = cdk.Environment(
    account=os.getenv('AWS_ACCOUNT_ID', os.getenv('CDK_DEFAULT_ACCOUNT')),
    region=os.getenv(
        'AWS_REGION', os.getenv('AWS_DEFAULT_REGION', os.getenv('CDK_DEFAULT_REGION'))
    ),
)

app = cdk.App()

github_stack = GitHubAuthStack(
    app, f'{PREFIX}GitHubAuthStack', env=env)

resources_stack = ResourcesStack(
    app, f'{PREFIX}ResourcesStack', env=env)

services_stack = ServicesStack(
    app, f'{PREFIX}ServicesStack', env=env,
    vpc=resources_stack.vpc,
    kms_key=resources_stack.kms_key,
    cluster=resources_stack.cluster,
    namespace=resources_stack.namespace,
    sessions_ddb_table=resources_stack.sessions_ddb_table,
)

app.synth()
