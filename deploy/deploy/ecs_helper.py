import os
from aws_cdk import (
    RemovalPolicy,
    Duration,
    CfnOutput,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_logs as logs,
    aws_elasticloadbalancingv2 as elbv2,
    aws_route53 as route53,
    aws_route53_targets as route53_targets,
    aws_certificatemanager as acm,
    aws_efs as efs,
    aws_iam as iam,
    aws_cloudwatch as cloudwatch,
    aws_applicationautoscaling as appscaling,
)
from dotenv import load_dotenv

load_dotenv('.env')
load_dotenv('../.env')

ALB_TIMEOUT = Duration.minutes(5)
MEMGRAPH_USER = os.getenv('MEMGRAPH_USER', '')
MEMGRAPH_PASSWORD = os.getenv('MEMGRAPH_PASSWORD', '')


class EcsHelper:
    def __init__(self, parent, prefix):
        self.parent = parent
        self.prefix = prefix

    @staticmethod
    def get_app_env_config(app_env_list):
        """
        Read sample config variables list and fill it with current env values
        """
        env_list = {}
        for key, value in app_env_list.items():
            if key in os.environ:
                env_list[key] = os.getenv(key)
            else:
                env_list[key] = value
        return env_list

    @staticmethod
    def get_app_env_config_with_prefix(app_env_list, prefix=""):
        """
        Read sample config variables list and fill it with current env values
        Prefix schema:
            If current env have "<prefix>Name", and we found "Name" in app_env_list, then set value to "Name"
        """
        env_list = {}
        for key, value in app_env_list.items():
            env_key = f"{prefix}{key}"
            if env_key in os.environ:
                env_list[key] = os.getenv(env_key)
            else:
                env_list[key] = value
        return env_list


    def create_efs_for_service(
        self, service_name, task_definition, container, vpc, sg, target_path,
        uid='1000',
        gid='1000',
        permissions='755',
    ):
        # Vars
        service_name_key = service_name[0].upper() + service_name[1:].replace('-', '')
        volume_name = f'{self.prefix}{service_name_key}TaskVolume'

        # Create an Amazon Elastic File System (EFS)
        file_system = efs.FileSystem(
            self.parent,
            f'{self.prefix}{service_name_key}Efs',
            vpc=vpc,
            security_group=sg,
            removal_policy=RemovalPolicy.DESTROY,
        )
        self.efs_file_system = file_system

        # Permissions
        file_system.add_to_resource_policy(
            iam.PolicyStatement(
                actions=['elasticfilesystem:ClientMount'],
                principals=[iam.AnyPrincipal()],
                conditions={
                    'Bool': {
                        'elasticfilesystem:AccessedViaMountTarget': 'true',
                    },
                },
            )
        )

        # EfsAccessPoint
        access_point = file_system.add_access_point(
            f'{self.prefix}{service_name_key}AccessPoint',
            path=target_path,
            create_acl=efs.Acl(
                owner_uid=uid,
                owner_gid=gid,
                permissions=permissions,
            ),
            posix_user=efs.PosixUser(
                uid=uid,
                gid=gid,
            ),
        )

        # EfsVolumeConfiguration
        volume_configuration = ecs.EfsVolumeConfiguration(
            file_system_id=file_system.file_system_id,
            transit_encryption='ENABLED',
            authorization_config=ecs.AuthorizationConfig(
                access_point_id=access_point.access_point_id,
                iam='ENABLED',
            ),
        )

        # Permissions
        task_definition.add_to_execution_role_policy(
            iam.PolicyStatement(
                actions=['elasticfilesystem:*'],
                resources=['*'],
                effect=iam.Effect.ALLOW,
            )
        )
        task_definition.add_to_task_role_policy(
            iam.PolicyStatement(
                actions=['elasticfilesystem:*'],
                resources=['*'],
                effect=iam.Effect.ALLOW,
            )
        )

        # Create
        task_definition.add_volume(
            name=volume_name,
            efs_volume_configuration=volume_configuration,
        )

        # Attach
        # Seems like need to be postponed and added AFTER creation
        container.add_mount_points(
            ecs.MountPoint(
                container_path=target_path,
                source_volume=volume_name,
                read_only=False,
            ),
        )

    def create_nlb_for_service(self, service_name, target, vpc, port, public_port=80):
        # Key
        service_name_key = service_name[0].upper() + service_name[1:].replace('-', '')

        # Network Load Balancer
        nlb = elbv2.NetworkLoadBalancer(
            self.parent, f'{self.prefix}{service_name_key}Nlb',
            vpc=vpc,
            internet_facing=True,
        )
        nlb.connections.allow_from_any_ipv4(
            ec2.Port.tcp(public_port),
            description='Allow HTTP access from anywhere',
        )
        nlb_lister = nlb.add_listener(
            f'{self.prefix}{service_name_key}NlbListener',
            port=public_port,
        )
        nlb_lister.add_targets(
            f'{self.prefix}{service_name_key}NlbTarget',
            port=port,
            targets=[target],
            health_check=None,  # Development
        )
        return nlb

    def create_alb_for_service(self, service_name, service, vpc, port, public_port=80):
        # Key
        service_name_key = service_name[0].upper() + service_name[1:].replace('-', '')

        # ALB
        alb = elbv2.ApplicationLoadBalancer(
            self.parent,
            f'{self.prefix}{service_name_key}Alb',
            vpc=vpc,
            internet_facing=True,
            idle_timeout=ALB_TIMEOUT,
        )
        alb.connections.allow_from_any_ipv4(
            ec2.Port.tcp(public_port),
            description='Allow HTTP access from anywhere',
        )
        # HTTP
        alb_listener_http = alb.add_listener(
            f'{self.prefix}{service_name_key}AlbHttpListener',
            port=public_port,
            open=True,
            protocol=elbv2.ApplicationProtocol.HTTP,
        )
        alb_listener_http.add_targets(
            f'{self.prefix}{service_name_key}HttpTarget',
            port=port,
            targets=[service],
            health_check=elbv2.HealthCheck(
                enabled=True,
                path='/',
                healthy_http_codes='200,404',
                interval=Duration.seconds(30),
                timeout=Duration.seconds(5),
                healthy_threshold_count=2,
                unhealthy_threshold_count=5,
            ),
        )
        return alb

    def create_service(
        self,
        vpc,
        cluster,
        image,
        env,
        role,
        kms_key,
        mem,
        cpu,
        port,
        service_name,
        idle_timeout=ALB_TIMEOUT,
        cert_arn=None,
        desired_count=1,
        health_check_grace_period=Duration.minutes(1),
    ):
        # Logs
        log_group = logs.LogGroup(
            self.parent,
            f'{self.prefix}{service_name}TaskLogs',
            log_group_name=f'/ecs/task/{self.prefix}{service_name}',
            encryption_key=kms_key,
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # Task
        task_definition = ecs.FargateTaskDefinition(
            self.parent,
            f'{self.prefix}{service_name}TaskDef',
            task_role=role,
            execution_role=role,
            memory_limit_mib=mem,
            cpu=cpu,
        )

        # Container
        container = task_definition.add_container(
            f'{self.prefix}{service_name}Container',
            image=image,
            environment=env,
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix=service_name,
                log_group=log_group,
            ),
        )

        # Open ports
        container.add_port_mappings(ecs.PortMapping(container_port=port))
        sg = ec2.SecurityGroup(
            self.parent,
            f'{self.prefix}{service_name}TaskSG',
            vpc=vpc,
            allow_all_outbound=True,
        )
        sg.add_ingress_rule(ec2.Peer.ipv4(vpc.vpc_cidr_block), ec2.Port.tcp(port))

        # Create a Fargate service to run the task
        service = ecs.FargateService(
            self.parent,
            f'{self.prefix}{service_name}Service',
            cluster=cluster,
            task_definition=task_definition,
            security_groups=[sg],
            assign_public_ip=True,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            desired_count=desired_count,
            min_healthy_percent=50,
            max_healthy_percent=200,
            health_check_grace_period=health_check_grace_period,
            capacity_provider_strategies=[
                # Less price
                ecs.CapacityProviderStrategy(
                    capacity_provider='FARGATE_SPOT',
                    weight=1,
                ),
                # Normal price
                ecs.CapacityProviderStrategy(
                    capacity_provider='FARGATE',
                    weight=0,
                ),
            ],
        )

        # Configure deployment circuit breaker for fast failure detection and rollback
        cfn_service = service.node.default_child
        cfn_service.deployment_configuration = ecs.CfnService.DeploymentConfigurationProperty(
            deployment_circuit_breaker=ecs.CfnService.DeploymentCircuitBreakerProperty(
                enable=True,
                rollback=True,
            ),
            maximum_percent=200,
            minimum_healthy_percent=50,
        )

        # Create ALB for service with public name
        alb = elbv2.ApplicationLoadBalancer(
            self.parent,
            f'{self.prefix}{service_name}Alb',
            vpc=vpc,
            internet_facing=True,
            idle_timeout=idle_timeout,
        )
        alb.connections.allow_from_any_ipv4(
            ec2.Port.tcp(80),
            description='Allow HTTP access from anywhere',
        )
        # HTTP
        alb_listener_http = alb.add_listener(
            f'{self.prefix}{service_name}AlbHttpListener',
            port=80,
            protocol=elbv2.ApplicationProtocol.HTTP,
            open=True,
        )
        alb_listener_http.add_targets(
            f'{self.prefix}{service_name}HttpTarget',
            port=port,
            protocol=elbv2.ApplicationProtocol.HTTP,
            targets=[service],
            health_check=elbv2.HealthCheck(
                enabled=True,
                path='/',
                healthy_http_codes='200,404',
                interval=Duration.seconds(30),
                timeout=Duration.seconds(5),
                healthy_threshold_count=2,
                unhealthy_threshold_count=5,
            ),
        )
        # HTTPS
        if cert_arn is not None and len(cert_arn) > 0:
            alb_listener_https = alb.add_listener(
                f'{self.prefix}{service_name}AlbHttpsListener',
                port=443,
                protocol=elbv2.ApplicationProtocol.HTTPS,
                open=True,
                certificates=[
                    elbv2.ListenerCertificate(
                        certificate_arn=cert_arn,
                    )
                ],
            )
            alb_listener_https.add_targets(
                f'{self.prefix}{service_name}HttpsTarget',
                port=port,
                protocol=elbv2.ApplicationProtocol.HTTP,
                targets=[service],
                health_check=elbv2.HealthCheck(
                    enabled=True,
                    path='/',
                    healthy_http_codes='200,404,401',
                    interval=Duration.seconds(30),
                    timeout=Duration.seconds(5),
                    healthy_threshold_count=2,
                    unhealthy_threshold_count=5,
                ),
            )
        return service, alb

    def get_hosted_zone(
        self,
        service_name,
        zone_name,
    ):
        # Hosted Zone
        hosted_zone = route53.HostedZone.from_lookup(
            self.parent,
            f'{self.prefix}{service_name}HostedZone',
            domain_name=zone_name,
        )
        return hosted_zone

    def create_cert(
        self,
        service_name,
        hosted_zone,
        domain_name,
        region=None,
    ):
        # Cert
        if region:
            cert = acm.DnsValidatedCertificate(
                self.parent,
                f'{self.prefix}{service_name}RegionDomainCert',
                domain_name=domain_name,
                hosted_zone=hosted_zone,
                region=region,
            )
        else:
            cert = acm.Certificate(
                self.parent,
                f'{self.prefix}{service_name}DomainCert',
                domain_name=domain_name,
                validation=acm.CertificateValidation.from_dns(hosted_zone),
            )
        return cert

    def create_domain(
        self,
        service_name,
        hosted_zone,
        domain_name,
        target_resource,
    ):
        # Domain
        domain = route53.ARecord(
            self.parent,
            f'{self.prefix}{service_name}Domain',
            zone=hosted_zone,
            record_name=domain_name,
            target=route53.RecordTarget.from_alias(target_resource),
            ttl=Duration.seconds(60),
        )
        return domain

    def create_domain_by_ip(
        self,
        service_name,
        hosted_zone,
        domain_name,
        target_ip,
    ):
        # Domain
        domain = route53.ARecord(
            self.parent,
            f'{self.prefix}{service_name}Domain',
            zone=hosted_zone,
            record_name=domain_name,
            target=route53.RecordTarget.from_ip_addresses(target_ip),
            ttl=Duration.seconds(60),
        )
        return domain

    def create_domain_by_target(
        self,
        service_name,
        hosted_zone,
        domain_name,
        target,
    ):
        # Domain
        domain = route53.ARecord(
            self.parent,
            f'{self.prefix}{service_name}Domain',
            zone=hosted_zone,
            record_name=domain_name,
            target=target,
            ttl=Duration.seconds(60),
        )
        return domain

    def create_service_with_domain(
        self,
        vpc,
        cluster,
        image,
        env,
        role,
        kms_key,
        mem,
        cpu,
        port,
        service_name,
        idle_timeout=Duration.minutes(5),
        cert_arn=None,
        zone_name=None,
        domain_name=None,
        desired_count=1,
        health_check_grace_period=Duration.minutes(1),
    ):
        # Domain
        hosted_zone = None
        alb_cert_arn = cert_arn
        if zone_name and domain_name:
            # Route53 Zone
            hosted_zone = self.get_hosted_zone(service_name, zone_name)
            # Cert
            cert = self.create_cert(service_name, hosted_zone, domain_name)
            alb_cert_arn = cert.certificate_arn

        # SERVICE
        service, alb = self.create_service(
            vpc,
            cluster,
            image,
            env,
            role,
            kms_key,
            mem,
            cpu,
            port,
            service_name,
            idle_timeout,
            cert_arn=alb_cert_arn,
            desired_count=desired_count,
            health_check_grace_period=health_check_grace_period,
        )

        # Domain
        if zone_name and domain_name:
            domain = self.create_domain(
                service_name,
                hosted_zone,
                domain_name,
                target_resource=route53_targets.LoadBalancerTarget(alb),
            )
            CfnOutput(
                self.parent,
                f'{self.prefix}{service_name}DomainName',
                value=domain.domain_name,
            )

        return service, alb


    def create_custom_service(
        self,
        service_name,
        cluster,
        image,
        env,
        ports,
        memory_limit_mib,
        cpu,
        sg,
        vpc,
        base_task_definition_role,
        namespace,
        kms_key,
        assign_public_ip=False,
        subnet_type=ec2.SubnetType.PUBLIC,
        additional_task_definition_processor=None,
        desired_count=1,
        use_spot=True,
        **kwargs,
    ) -> tuple[ecs.FargateService, ecs.FargateTaskDefinition, ecs.ContainerDefinition]:
        # Host
        service_name_key = service_name[0].upper() + service_name[1:].replace('-', '')

        # Logs
        log_group = logs.LogGroup(
            self.parent,
            f'{self.prefix}{service_name_key}TaskLogs',
            log_group_name=f'/ecs/{self.prefix}{service_name_key}TaskLogs',
            encryption_key=kms_key,
            retention=logs.RetentionDays.ONE_WEEK,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # Task
        task_definition = ecs.FargateTaskDefinition(
            self.parent,
            f'{self.prefix}{service_name_key}TaskDef',
            task_role=base_task_definition_role,
            execution_role=base_task_definition_role,
            memory_limit_mib=memory_limit_mib,
            cpu=cpu,
        )

        # Container
        container = task_definition.add_container(
            f'{self.prefix}{service_name_key}Container',
            image=image,
            environment=env,
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix=service_name,
                log_group=log_group,
            ),
        )

        # Open ports
        for port in ports:
            container.add_port_mappings(ecs.PortMapping(container_port=port))
            sg.add_ingress_rule(
                ec2.Peer.ipv4(vpc.vpc_cidr_block)
                if vpc is not None
                else ec2.Peer.any_ipv4(),
                ec2.Port.tcp(port),
            )

        # Policy
        if additional_task_definition_processor is not None:
            additional_task_definition_processor(task_definition, container)

        # Create a Fargate service to run the task
        service = ecs.FargateService(
            self.parent,
            f'{self.prefix}{service_name_key}Service',
            service_name=f'{self.prefix}{service_name_key}Service',
            cluster=cluster,
            task_definition=task_definition,
            security_groups=[sg],
            assign_public_ip=assign_public_ip,
            cloud_map_options=ecs.CloudMapOptions(
                name=service_name,
                cloud_map_namespace=namespace,
            ),
            vpc_subnets=ec2.SubnetSelection(subnet_type=subnet_type),
            desired_count=desired_count,
            min_healthy_percent=0,
            max_healthy_percent=100,
            capacity_provider_strategies=[
                # Less price
                ecs.CapacityProviderStrategy(
                    capacity_provider='FARGATE_SPOT',
                    weight=1 if use_spot else 0,
                ),
                # Normal price
                ecs.CapacityProviderStrategy(
                    capacity_provider='FARGATE',
                    weight=0 if use_spot else 1,
                ),
            ],
        )

        # Configure deployment circuit breaker for fast failure detection and rollback
        cfn_service = service.node.default_child
        cfn_service.deployment_configuration = ecs.CfnService.DeploymentConfigurationProperty(
            deployment_circuit_breaker=ecs.CfnService.DeploymentCircuitBreakerProperty(
                enable=True,
                rollback=True,
            ),
            maximum_percent=100,
            minimum_healthy_percent=0,
        )

        return (service, task_definition, container)

    def create_qdrant(
        self,
        cluster,
        vpc,
        kms_key,
        namespace,
    ) -> str:
        # Security Group
        sg = ec2.SecurityGroup(self.parent, f'{self.prefix}EcsSecGroup', vpc=vpc)
        sg.add_ingress_rule(ec2.Peer.any_ipv4(), ec2.Port.tcp(2049))  # efs

        # Role
        task_definition_role = iam.Role(
            self.parent,
            f'{self.prefix}QdrantTaskDefinitionRole',
            assumed_by=iam.CompositePrincipal(
                iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
                iam.ServicePrincipal('ecs.amazonaws.com'),
            ),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    'service-role/AmazonECSTaskExecutionRolePolicy'
                ),
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    'AmazonSSMReadOnlyAccess'
                ),
            ],
        )

        # Qdrant
        qdrant_service_name = 'qdrant'
        qdrant_host = f'{qdrant_service_name}.{namespace.namespace_name}'

        def qdrant_efs_processor(task_def, container):
            self.create_efs_for_service(
                service_name=qdrant_service_name,
                task_definition=task_def,
                container=container,
                vpc=vpc,
                sg=sg,
                target_path='/qdrant/storage',
            )

        # Service
        qdrant_service, qdrant_task_definition, qdrant_container = self.create_custom_service(
            service_name=qdrant_service_name,
            cluster=cluster,
            image=ecs.ContainerImage.from_registry('qdrant/qdrant:latest'),
            env={
                'QDRANT__LOG_LEVEL': 'INFO',
            },
            ports=[6333, 6334, 6335],
            memory_limit_mib=1024,
            cpu=256,
            sg=sg,
            vpc=vpc,
            namespace=namespace,
            kms_key=kms_key,
            base_task_definition_role=task_definition_role,
            additional_task_definition_processor=qdrant_efs_processor,
            assign_public_ip=False,
            subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
        )
        self.qdrant_service = qdrant_service
        self.qdrant_task_definition = qdrant_task_definition
        self.qdrant_container = qdrant_container
        return qdrant_host

    def create_memgraph(
        self,
        cluster,
        vpc,
        kms_key,
        namespace,
        memory,
        cpu,
        is_public=False,
        service_name='Memgraph',
    ) -> str:
        # Security Group
        sg = ec2.SecurityGroup(self.parent, f'{self.prefix}{service_name}EcsSecGroup', vpc=vpc)
        sg.add_ingress_rule(ec2.Peer.any_ipv4(), ec2.Port.tcp(2049))  # efs

        # Role
        task_definition_role = iam.Role(
            self.parent,
            f'{self.prefix}{service_name}TaskDefinitionRole',
            assumed_by=iam.CompositePrincipal(
                iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
                iam.ServicePrincipal('ecs.amazonaws.com'),
            ),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    'service-role/AmazonECSTaskExecutionRolePolicy'
                ),
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    'AmazonSSMReadOnlyAccess'
                ),
            ],
        )
        # Add KMS policy
        task_definition_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    'kms:Decrypt',
                    'kms:Encrypt',
                    'kms:GenerateDataKey',
                ],
                resources=[kms_key.key_arn],
            ),
        )

        # Memgraph
        memgraph_host = f'{service_name.lower()}.{namespace.namespace_name}'

        def memgraph_efs_processor(task_def, container):
            # Attach EFS
            self.create_efs_for_service(
                service_name=service_name,
                task_definition=task_def,
                container=container,
                vpc=vpc,
                sg=sg,
                target_path='/var/lib/memgraph',
                # target_path='/var/lib/memgraph/persistent_data',
                # uid='101',  # uid=101(memgraph)
                uid='0',  # root
                # gid='103',  # gid=103(memgraph) groups=103(memgraph)
                gid='0',  # root
                permissions='777',
            )
            # Patch 'ulimits'
            # Based on https://memgraph.com/docs/database-management/system-configuration#with-docker
            # 8GB-32GB -> vm.max_map_count=262144
            container.add_ulimits(
                ecs.Ulimit(name=ecs.UlimitName.NOFILE, soft_limit=65535, hard_limit=65535),
            )

        # Service
        memgraph_service, memgraph_task_definition, memgraph_container = self.create_custom_service(
            service_name=service_name,
            cluster=cluster,
            # image=ecs.ContainerImage.from_registry('memgraph/memgraph-mage:3.4'),
            # image=ecs.ContainerImage.from_asset(directory='memgraph',file='Dockerfile'),
            image=ecs.ContainerImage.from_asset(directory='memgraph',file='root.Dockerfile'),
            # image=ecs.ContainerImage.from_asset(directory='memgraph',file='memgraph.Dockerfile'),
            env={
                'MEMGRAPH': (
                    '--log-level=TRACE '
                    '--storage-properties-on-edges=true '
                    '--storage-snapshot-interval-sec=300 '
                    '--storage-snapshot-retention-count=5 '
                    '--storage-snapshot-on-exit=true '
                    '--storage-mode=IN_MEMORY_TRANSACTIONAL '
                    # '--storage-mode=ON_DISK_TRANSACTIONAL '
                    # '--storage-mode=IN_MEMORY_TRANSACTIONAL '
                    # '--data-directory=/var/lib/memgraph/persistent_data '
                ),
                'MEMGRAPH_USER': MEMGRAPH_USER,
                'MEMGRAPH_PASSWORD': MEMGRAPH_PASSWORD,
            },
            ports=[3000, 7444, 7687],
            memory_limit_mib=memory,
            cpu=cpu,
            sg=sg,
            vpc=vpc,
            namespace=namespace,
            kms_key=kms_key,
            base_task_definition_role=task_definition_role,
            additional_task_definition_processor=memgraph_efs_processor,
            assign_public_ip=is_public,
            subnet_type=ec2.SubnetType.PUBLIC if is_public else ec2.SubnetType.PRIVATE_WITH_EGRESS,
            use_spot=False,  # more stable
        )

        # Add ingress from public subnet, if public
        if is_public:
            sg.add_ingress_rule(ec2.Peer.any_ipv4(), ec2.Port.tcp(7687))
            sg.add_ingress_rule(ec2.Peer.any_ipv4(), ec2.Port.tcp(3000))
            sg.add_ingress_rule(ec2.Peer.any_ipv4(), ec2.Port.tcp(7444))

        self.memgraph_service = memgraph_service
        self.memgraph_task_definition = memgraph_task_definition
        self.memgraph_container = memgraph_container
        return memgraph_host

    def create_memgraph_lab_with_domain(
        self,
        cluster,
        vpc,
        kms_key,
        memgraph_host,
        domain_name,
        zone_name,
        service_name = 'MemgraphLabApp',
    ) -> str:
        # Role
        task_definition_role = iam.Role(
            self.parent,
            f'{self.prefix}{service_name}TaskDefinitionRole',
            assumed_by=iam.CompositePrincipal(
                iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
                iam.ServicePrincipal('ecs.amazonaws.com'),
            ),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    'service-role/AmazonECSTaskExecutionRolePolicy'
                ),
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    'AmazonSSMReadOnlyAccess'
                ),
            ],
        )
        # Add KMS policy
        task_definition_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    'kms:Decrypt',
                    'kms:Encrypt',
                    'kms:GenerateDataKey',
                ],
                resources=[kms_key.key_arn],
            ),
        )

        # Service
        memgraph_lab_service, memgraph_lab_alb = self.create_service_with_domain(
            vpc=vpc,
            cluster=cluster,
            image=ecs.ContainerImage.from_registry('memgraph/lab:latest'),
            env={
                'QUICK_CONNECT_MG_HOST': memgraph_host,
                'QUICK_CONNECT_MG_PORT': '7687',
            } if memgraph_host else {},
            role=task_definition_role,
            kms_key=kms_key,
            mem=1024,
            cpu=256,
            port=3000,
            service_name=service_name,
            zone_name=zone_name,
            domain_name=domain_name,
        )
        self.memgraph_lab_service = memgraph_lab_service
        self.memgraph_lab_alb = memgraph_lab_alb
        return domain_name

    def create_memgraph_lab(
        self,
        cluster,
        vpc,
        kms_key,
        namespace,
        memgraph_host,
        service_name = 'MemgraphLab',
    ) -> str:
        sg = ec2.SecurityGroup(self.parent, f'{self.prefix}{service_name}EcsSecGroup', vpc=vpc)

        # Role
        task_definition_role = iam.Role(
            self.parent,
            f'{self.prefix}{service_name}TaskDefinitionRole',
            assumed_by=iam.CompositePrincipal(
                iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
                iam.ServicePrincipal('ecs.amazonaws.com'),
            ),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    'service-role/AmazonECSTaskExecutionRolePolicy'
                ),
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    'AmazonSSMReadOnlyAccess'
                ),
            ],
        )
        # Add KMS policy
        task_definition_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    'kms:Decrypt',
                    'kms:Encrypt',
                    'kms:GenerateDataKey',
                ],
                resources=[kms_key.key_arn],
            ),
        )

        # Memgraph
        memgraph_lab_host = f'{service_name.lower()}.{namespace.namespace_name}'

        # Service
        memgraph_lab_service, memgraph_lab_task_definition, memgraph_lab_container = self.create_custom_service(
            service_name=service_name,
            cluster=cluster,
            # image=ecs.ContainerImage.from_asset(directory='memgraph',file='root.Dockerfile'),
            image=ecs.ContainerImage.from_registry('memgraph/lab:latest'),
            env={
                'QUICK_CONNECT_MG_HOST': memgraph_host,
                'QUICK_CONNECT_MG_PORT': '7687',
            } if memgraph_host else {},
            ports=[3000],
            memory_limit_mib=1024,
            cpu=256,
            sg=sg,
            vpc=vpc,
            namespace=namespace,
            kms_key=kms_key,
            base_task_definition_role=task_definition_role,
            assign_public_ip=True,
            subnet_type=ec2.SubnetType.PUBLIC,
        )

        # Add ingress: port 3000, source: *
        sg.add_ingress_rule(ec2.Peer.any_ipv4(), ec2.Port.tcp(3000))

        self.memgraph_lab_service: ecs.FargateService = memgraph_lab_service
        self.memgraph_lab_task_definition: ecs.FargateTaskDefinition = memgraph_lab_task_definition
        self.memgraph_lab_container: ecs.ContainerDefinition = memgraph_lab_container
        return memgraph_lab_host

    def apply_scaling(
        self,
        service_name,
        service,
        role,
        cluster,
        alb,
        max_capacity=4,
        min_capacity=1,
    ) -> None:
        # Scaling
        # ECS Service Auto Scaling
        scalable_target = appscaling.ScalableTarget(
            self.parent,
            f'{self.prefix}{service_name}ScalableTarget',
            service_namespace=appscaling.ServiceNamespace.ECS,
            max_capacity=max_capacity,
            min_capacity=min_capacity,
            resource_id=f'service/{cluster.cluster_name}/{service.service_name}',
            scalable_dimension='ecs:service:DesiredCount',
            role=role,  # Use the ECS task role for scaling
        )

        # Custom metric for requests per target. '1' means all nodes get request every second.
        alb_metric = cloudwatch.MathExpression(
            expression=f"SELECT AVG(RequestCountPerTarget) FROM SCHEMA(\"AWS/ApplicationELB\", LoadBalancer,TargetGroup) WHERE LoadBalancer = '{alb.load_balancer_full_name}'",
            using_metrics={},  # No direct metrics referenced, as this is a query
            label="RequestCountPerTarget",
            period=Duration.minutes(1),
        )

        # Step scaling
        scalable_target.scale_on_metric(
            f'{self.prefix}{service_name}StepScalingPolicy',
            metric=alb_metric,
            scaling_steps=[
                appscaling.ScalingInterval(upper=0.10, change=-1),  # Scale in if <10%
                appscaling.ScalingInterval(lower=0.30, change=+1),  # Scale out if >30%
            ],
            adjustment_type=appscaling.AdjustmentType.CHANGE_IN_CAPACITY,
            cooldown=Duration.minutes(15),
            evaluation_periods=3,   # 3 minutes
            datapoints_to_alarm=2,  # 2 minutes
        )
