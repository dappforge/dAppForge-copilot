from aws_cdk import (
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_efs as efs,
    RemovalPolicy,
)


class EC2Helper:
    def __init__(self, parent, prefix):
        self.parent = parent
        self.prefix = prefix

    def get_ami(self):
        # Canonical, Ubuntu, 24.04, amd64 noble image
        ami_id = 'ami-020cba7c55df1f615' # ubuntu 24.04 at us-east-1
        return ami_id

    def create_eip(self):
        # Static IP
        elastic_ip = ec2.CfnEIP(self.parent, f'{self.prefix}ElasticIP')
        return elastic_ip

    def create_efs(self, vpc):
        # SG
        sg = ec2.SecurityGroup(self.parent, f'{self.prefix}EfsSecGroup', vpc=vpc)
        sg.add_ingress_rule(ec2.Peer.any_ipv4(), ec2.Port.tcp(2049))  # efs

        # EFS
        efs_file_system = efs.FileSystem(
            self.parent,
            f'{self.prefix}EFS',
            vpc=vpc,
            security_group=sg,
            removal_policy=RemovalPolicy.RETAIN,
        )
        self.efs_file_system = efs_file_system

        # Permissions
        efs_file_system.add_to_resource_policy(
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
        return efs_file_system

    def create_ec2_instance(
        self,
        name,
        ports,
        key_pair_name,
        vpc,
        kms_key,
        script_path='startup.sh',
        subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
        instance_type=ec2.InstanceType.of(
            ec2.InstanceClass.T2,
            ec2.InstanceSize.LARGE,
        ),
        volume_size=10,
        eip=None,
    ) -> ec2.Instance:
        # Canonical, Ubuntu, 24.04, amd64 noble image
        ami_id = self.get_ami()

        # EC2 Security Group
        ec2_security_group = ec2.SecurityGroup(
            self.parent,
            f'{self.prefix}{name}SecurityGroup',
            vpc=vpc,
            allow_all_outbound=True,
            description=f'{self.prefix}{name} Security Group',
        )
        # Allow ingress ports
        for port in ports:
            ec2_security_group.add_ingress_rule(
                ec2.Peer.any_ipv4(),
                ec2.Port.tcp(port),
            )

        # EC2 Role
        ec2_role = iam.Role(
            self.parent,
            f'{self.prefix}{name}Role',
            assumed_by=iam.ServicePrincipal('ec2.amazonaws.com'),
        )
        # Add KMS access
        ec2_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    'kms:*',
                ],
                resources=[kms_key.key_arn],
            )
        )
        # Add CloudWatch access
        ec2_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    'cloudwatch:PutMetricData',
                    'cloudwatch:GetMetricData',
                    'cloudwatch:GetMetricStatistics',
                    'cloudwatch:ListMetrics',
                    'logs:DescribeLogGroups',
                    'logs:GetLogGroupFields',
                    'logs:StartQuery',
                    'logs:StopQuery',
                    'logs:GetQueryResults',
                    'logs:GetLogEvents',
                ],
                resources=['*'],
            )
        )
        self.ec2_role = ec2_role

        # EC2 Instance t2.large
        ec2_instance = ec2.Instance(
            self.parent,
            f'{self.prefix}{name}Instance',
            instance_name=f'{self.prefix}{name}',
            key_pair=ec2.KeyPair.from_key_pair_name(
                self.parent,
                f'{self.prefix}{name}KeyPair',
                key_pair_name,
            ),
            # Machine type
            instance_type=instance_type,
            machine_image=ec2.MachineImage.generic_linux({
                'us-east-1': ami_id,
            }),
            # Public VPC and IP
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=subnet_type,
            ),
            associate_public_ip_address=True if subnet_type == ec2.SubnetType.PUBLIC else False,
            # Storage
            block_devices=[
                ec2.BlockDevice(
                    device_name='/dev/sda1',
                    volume=ec2.BlockDeviceVolume.ebs(
                        volume_size=volume_size,
                        encrypted=True,
                        kms_key=kms_key,
                        delete_on_termination=True,
                    ),
                ),
            ],
            # Permissions
            security_group=ec2_security_group,
            role=ec2_role,
        )
        self.ec2_instance = ec2_instance

        # Startup script
        user_data_script = open(script_path, 'r').read()
        if self.efs_file_system:
            # Replace EFS_FS_ID with the actual EFS ID
            user_data_script = user_data_script.replace('EFS_FS_ID', self.efs_file_system.file_system_id)
        ec2_instance.add_user_data(user_data_script)

        if eip:
            # Attach EIP to EC2
            ec2.CfnEIPAssociation(
                self.parent,
                f'{self.prefix}{name}EIPAssociation',
                instance_id=ec2_instance.instance_id,
                eip=eip.ref,
            )

        return ec2_instance
