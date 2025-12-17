import os
from aws_cdk import (
    Stack,
    Duration,
    aws_ec2 as ec2,
    aws_ecs as ecs,    
    aws_kms as kms,
    aws_servicediscovery as servicediscovery,
    CfnOutput,
    aws_iam as iam,
    aws_route53 as route53,
    aws_route53_targets as route53_targets,
    aws_dynamodb as dynamodb,
)
from constructs import Construct
from dotenv import load_dotenv

load_dotenv('.env')
load_dotenv('../.env')

from deploy.ecs_helper import EcsHelper
from deploy.ec2_helper import EC2Helper

PREFIX = os.getenv('PREFIX', '')
ZONE_NAME = os.getenv('ZONE_NAME')
API_DOMAIN_NAME = os.getenv('API_DOMAIN_NAME')
BASTION_DOMAIN_NAME = os.getenv('BASTION_DOMAIN_NAME')
KEY_PAIR_NAME = os.getenv('KEY_PAIR_NAME', 'dapp-nl-dev-instance-keys')
MEMGRAPH_USER = os.getenv('MEMGRAPH_USER', '')
MEMGRAPH_PASSWORD = os.getenv('MEMGRAPH_PASSWORD', '')

INTERNAL_LANGFUSE_PUBLIC_KEY = os.getenv('INTERNAL_LANGFUSE_PUBLIC_KEY', '')
INTERNAL_LANGFUSE_SECRET_KEY = os.getenv('INTERNAL_LANGFUSE_SECRET_KEY', '')
INTERNAL_LANGFUSE_HOST = os.getenv('INTERNAL_LANGFUSE_HOST', 'https://cloud.langfuse.com')
CLIENT_LANGFUSE_PUBLIC_KEY = os.getenv('CLIENT_LANGFUSE_PUBLIC_KEY', '')
CLIENT_LANGFUSE_SECRET_KEY = os.getenv('CLIENT_LANGFUSE_SECRET_KEY', '')
CLIENT_LANGFUSE_HOST = os.getenv('CLIENT_LANGFUSE_HOST', 'https://cloud.langfuse.com')


class ServicesStack(Stack):
    def create_bastion(self) -> None:
        # Bastion to debug Memgraph
        bastion_sg = ec2.SecurityGroup(
            self,
            'BastionSG',
            vpc=self.vpc,
            allow_all_outbound=True,
        )
        bastion_sg.add_ingress_rule(
            ec2.Peer.any_ipv4(),
            ec2.Port.tcp(22),
            'Allow SSH from anywhere',
        )
        bastion = ec2.Instance(
            self,
            'Bastion',
            instance_name=f'{PREFIX}Bastion',
            instance_type=ec2.InstanceType.of(
                ec2.InstanceClass.T3,
                ec2.InstanceSize.MICRO,
            ),
            machine_image=ec2.MachineImage.latest_amazon_linux2(),
            vpc=self.vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            security_group=bastion_sg,
            key_pair=ec2.KeyPair.from_key_pair_name(self, 'BastionKeyPair', KEY_PAIR_NAME),
        )
        self.bastion = bastion
        CfnOutput(self, 'BastionPublicIp', value=bastion.instance_public_ip)

        # Bastion Domain Alias
        if ZONE_NAME and BASTION_DOMAIN_NAME:
            ecs_helper = EcsHelper(self, PREFIX)
            hosted_zone = ecs_helper.get_hosted_zone(
                service_name='Bastion',
                zone_name=ZONE_NAME,
            )
            ecs_helper.create_domain_by_ip(
                service_name='Bastion',
                hosted_zone=hosted_zone,
                domain_name=BASTION_DOMAIN_NAME,
                target_ip=bastion.instance_public_ip,
            )
            CfnOutput(self, 'BastionDomainName', value=BASTION_DOMAIN_NAME)

    def create_memgraph_db(self) -> None:
        # Ecs Helper
        ecs_helper = EcsHelper(self, PREFIX)
        # Create Memgraph from "memgraph/memgraph-mage"
        memgraph_names = [
            {'name': 'Solidity', 'cpu': 1024, 'memory': 1024*4},
            {'name': 'Ink', 'cpu': 256, 'memory': 1024*2},
            {'name': 'Rust', 'cpu': 1024, 'memory': 1024*4},
            {'name': 'Substrate', 'cpu': 256, 'memory': 1024*2},
        ]
        memgraph_hosts = []
        # Create Memgraph
        for memgraph_info in memgraph_names:
            memgraph_host = ecs_helper.create_memgraph(
                cluster=self.cluster,
                vpc=self.vpc,
                kms_key=self.kms_key,
                namespace=self.namespace,
                # https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task-cpu-memory-error.html
                memory=memgraph_info["memory"],
                cpu=memgraph_info["cpu"],
                service_name=f'Memgraph{memgraph_info["name"]}',
                is_public=True,  # TODO: Public only for debugging, make it private after
            )
            memgraph_hosts.append(memgraph_host)
            memgraph_info['host'] = memgraph_host
            CfnOutput(self, f'Memgraph{memgraph_info["name"]}Host', value=memgraph_host)
        self.memgraph_hosts = memgraph_hosts
        self.memgraph_infos = memgraph_names

    def create_memgraph_as_ec2(self) -> None:
        # Create EC2 Helper
        ec2_helper = EC2Helper(self, PREFIX)
        # Create EFS
        efs_file_system = ec2_helper.create_efs(self.vpc)
        self.efs_file_system = efs_file_system
        CfnOutput(self, f'{PREFIX}MemgraphEfsFileSystem', value=efs_file_system.file_system_id)
        # Create EC2 Instance for Memgraph
        ec2_instance = ec2_helper.create_ec2_instance(
            name='Memgraph',
            ports=[22, 7444, 7687, 3000],
            key_pair_name=KEY_PAIR_NAME,
            vpc=self.vpc,
            kms_key=self.kms_key,
            script_path='memgraph/user-script.sh',
            subnet_type=ec2.SubnetType.PUBLIC,
            instance_type=ec2.InstanceType.of(
                # t3.medium (4gb, 2vcpu, 30usd/month)
                ec2.InstanceClass.T3,
                ec2.InstanceSize.MEDIUM,
            ),
            volume_size=10,
            eip=None,
        )
        ec2_instance.node.add_dependency(efs_file_system)  # Ensure EFS is created before EC2
        self.ec2_instance = ec2_instance
        CfnOutput(self, 'MemgraphHost', value=ec2_instance.instance_private_ip)

    def create_memgraph_lab(self) -> None:
        # Ecs Helper
        ecs_helper = EcsHelper(self, PREFIX)
        # Create MemgraphLab from "memgraph/lab"
        ecs_helper.create_memgraph_lab_with_domain(
            cluster=self.cluster,
            vpc=self.vpc,
            kms_key=self.kms_key,
            memgraph_host=None,
            domain_name=None,
            zone_name=None,
        )
        memgraph_lab_alb: ecs.ApplicationLoadBalancer = ecs_helper.memgraph_lab_alb
        CfnOutput(self, 'MemgraphLabDomainName', value=memgraph_lab_alb.load_balancer_dns_name)

    def create_api_service(self) -> None:
        ecs_helper = EcsHelper(self, PREFIX)
        # Role
        api_role = iam.Role(
            self,
            f'{PREFIX}ApiRole',
            assumed_by=iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name('service-role/AmazonECSTaskExecutionRolePolicy'),
                iam.ManagedPolicy.from_aws_managed_policy_name('AmazonSSMReadOnlyAccess'),
            ],
        )
        # Add Bedrock Policy
        api_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock:ListFoundationModels",
                    "bedrock:InvokeModelWithResponseStream",
                    "bedrock:InvokeModel",
                    "bedrock:GetFoundationModel",
                ],
                resources=[
                    "arn:aws:bedrock:*::foundation-model/*",
                    f"arn:aws:bedrock:*:{self.account}:*",
                ],
            )
        )
        # Add DDB Policy
        api_role.add_to_policy(
            iam.PolicyStatement(
                actions=['dynamodb:*'],
                resources=[self.sessions_ddb_table.table_arn],
            )
        )

        # Image
        api_image = ecs.ContainerImage.from_asset(
            directory='..',  # Root directory of the project
            file='Dockerfile',
        )
        # Memgraph DB Envs
        memgraph_envs = {
            f'MEMGRAPH_{info["name"].upper()}_HOST': info["host"] for info in self.memgraph_infos
        } if self.memgraph_infos else {}

        # Create Service
        api_service, api_alb = ecs_helper.create_service_with_domain(
            vpc=self.vpc,
            cluster=self.cluster,
            image=api_image,
            env={
                **memgraph_envs,
                'MEMGRAPH_USER': MEMGRAPH_USER,
                'MEMGRAPH_PASSWORD': MEMGRAPH_PASSWORD,
                'MEMGRAPH_PORT': '7687',
                'DDB_CHAT_TABLE': self.sessions_ddb_table.table_name,
                'INTERNAL_LANGFUSE_PUBLIC_KEY': INTERNAL_LANGFUSE_PUBLIC_KEY,
                'INTERNAL_LANGFUSE_SECRET_KEY': INTERNAL_LANGFUSE_SECRET_KEY,
                'INTERNAL_LANGFUSE_HOST': INTERNAL_LANGFUSE_HOST,
                'CLIENT_LANGFUSE_PUBLIC_KEY': CLIENT_LANGFUSE_PUBLIC_KEY,
                'CLIENT_LANGFUSE_SECRET_KEY': CLIENT_LANGFUSE_SECRET_KEY,
                'CLIENT_LANGFUSE_HOST': CLIENT_LANGFUSE_HOST,
                'REDIS_HOST': 'localhost',
                'REDIS_PORT': '6379',
            },
            role=api_role,
            kms_key=self.kms_key,
            # https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task-cpu-memory-error.html
            mem=1024*2,
            cpu=1024,
            port=8081,
            service_name='Api',
            zone_name=ZONE_NAME,
            domain_name=API_DOMAIN_NAME,
            health_check_grace_period=Duration.minutes(5),
        )
        self.api_service = api_service
        self.api_alb = api_alb
        CfnOutput(self, 'ApiDomainName', value=api_alb.load_balancer_dns_name)

        # Apply scaling
        ecs_helper.apply_scaling(
            service_name='Api',
            service=api_service,
            role=api_role,
            cluster=self.cluster,
            alb=api_alb,
            max_capacity=5,
            min_capacity=1,
        )

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        # Args
        self.vpc: ec2.Vpc = kwargs.pop('vpc')
        self.kms_key: kms.Key = kwargs.pop('kms_key')
        self.cluster: ecs.Cluster = kwargs.pop('cluster')
        self.namespace: servicediscovery.PrivateDnsNamespace = kwargs.pop('namespace')
        self.sessions_ddb_table: dynamodb.Table = kwargs.pop('sessions_ddb_table')

        # Super
        super().__init__(scope, construct_id, **kwargs)

        # Bastion
        # TODO: Can be removed after debugging
        # self.create_bastion()

        # Memgraph
        self.create_memgraph_db()

        # MemgraphLab
        self.create_memgraph_lab()

        # API Service with Balancing
        self.create_api_service()
