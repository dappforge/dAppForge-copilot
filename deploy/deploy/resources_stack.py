import os
from aws_cdk import (
    Stack,
    RemovalPolicy,
    aws_ec2 as ec2,
    aws_kms as kms,
    aws_iam as iam,
    aws_ecs as ecs,
    aws_servicediscovery as servicediscovery,
    aws_dynamodb as dynamodb,
)
from constructs import Construct
from dotenv import load_dotenv

load_dotenv('.env')
load_dotenv('../.env')

PREFIX = os.getenv('PREFIX', '')
VPC_ID = os.getenv('VPC_ID')
VPC_DEFAULT = os.getenv('VPC_DEFAULT', 'false').lower() == 'true'


class ResourcesStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Create VPC or use existing VPC
        if VPC_DEFAULT:
            vpc = ec2.Vpc.from_lookup(
                self,
                f'{PREFIX}Vpc',
                is_default=True,
            )
        elif VPC_ID:
            vpc = ec2.Vpc.from_lookup(
                self,
                f'{PREFIX}Vpc',
                vpc_id=VPC_ID,
            )
        else:
            vpc = ec2.Vpc(
                self,
                f'{PREFIX}Vpc',
                max_azs=2,
                subnet_configuration=[
                    ec2.SubnetConfiguration(
                        name='public',
                        subnet_type=ec2.SubnetType.PUBLIC,
                    ),
                    ec2.SubnetConfiguration(
                        name='private',
                        subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    ),
                ],
                create_internet_gateway=True,
                nat_gateways=1,
            )
        self.vpc = vpc

        # KMS Key
        kms_key = kms.Key(
            self,
            'KmsKey',
            removal_policy=RemovalPolicy.DESTROY,
        )
        kms_key.add_to_resource_policy(
            iam.PolicyStatement(
                actions=[
                    'kms:Encrypt',
                    'kms:Decrypt',
                    'kms:ReEncrypt*',
                    'kms:GenerateDataKey*',
                    'kms:DescribeKey',
                ],
                resources=['*'],
                principals=[iam.ServicePrincipal('logs.amazonaws.com')],
            )
        )
        self.kms_key = kms_key

        # ECS Cluster
        cluster = ecs.Cluster(
            self,
            f"{PREFIX}Cluster",
            cluster_name=f"{PREFIX}SolutionCluster",
            vpc=vpc,
        )
        self.cluster = cluster

        # Namespace for service discovery
        namespace = servicediscovery.PrivateDnsNamespace(
            self,
            f"{PREFIX}Namespace",
            name=f"{PREFIX.lower()}-service.local",
            vpc=vpc,
        )
        self.namespace = namespace

        # DynamoDB Table for Sessions
        sessions_ddb_table = dynamodb.Table(
            self,
            f"{PREFIX}SessionsDynamoDBTable",
            table_name=f"{PREFIX}ChatSessionsTable",
            partition_key=dynamodb.Attribute(name="id", type=dynamodb.AttributeType.STRING),
            removal_policy=RemovalPolicy.DESTROY,
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
        )
        self.sessions_ddb_table = sessions_ddb_table
