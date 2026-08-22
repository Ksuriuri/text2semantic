#!/bin/bash
# The box spot_supervisor.py runs on, and the only way into the training nodes.
#
# Both of those follow from one decision. A full-width p5 needs 32 network
# interfaces, EC2 refuses a public IPv4 to an instance with more than one, so the
# nodes are private-only: nothing outside this VPC can reach them, and the
# supervisor's repair loop is plain ssh to a private address. So it has to live
# here, in the public subnet of the same VPC, on the same security group the nodes
# use (which allows everything within itself).
#
# It gets its own role rather than the nodes' NoizT2sStagingRole, because this is
# the only thing in the run that has to create instances, and that is not a
# permission the training nodes should carry.
#
# t3.small: the loop is one describe every 60 s. It is also the jump host, so it
# outlives the run by as long as diagnosis takes -- terminate it with the NAT
# gateway at the end.
set -euo pipefail

REGION=us-east-2
BUCKET=noiz-t2s-us-east-2
ROLE=NoizT2sSupervisorRole
PROFILE=NoizT2sSupervisorProfile
NAME=noiz-t2s-supervisor
KEY=noiz-t2s-20260819
SG=sg-07840151fef77b3fe
SUBNET=subnet-0d364f8ca36b5d85b   # public, us-east-2a

TRUST='{"Version":"2012-10-17","Statement":[{"Effect":"Allow",
  "Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
# RunInstances is left unscoped by resource on purpose. Resource-level conditions
# on it cover a dozen implicit resources (interfaces, volumes, the template, the
# placement group), and a policy gap there does not fail at launch -- it fails at
# 3am when a preemption needs repairing. The region condition and a role nothing
# else assumes are the boundary instead.
POLICY=$(cat <<'JSON'
{"Version":"2012-10-17","Statement":[
 {"Sid":"Fleet","Effect":"Allow","Action":[
   "ec2:DescribeInstances","ec2:DescribeInstanceStatus","ec2:DescribeLaunchTemplates",
   "ec2:DescribeLaunchTemplateVersions","ec2:DescribeSpotInstanceRequests",
   "ec2:DescribeSubnets","ec2:DescribeNetworkInterfaces",
   "ec2:RunInstances","ec2:StartInstances","ec2:StopInstances",
   "ec2:TerminateInstances","ec2:CreateTags"],
  "Resource":"*","Condition":{"StringEquals":{"aws:RequestedRegion":"us-east-2"}}},
 {"Sid":"PassNodeRole","Effect":"Allow","Action":"iam:PassRole",
  "Resource":"arn:aws:iam::*:role/NoizT2sStagingRole",
  "Condition":{"StringEquals":{"iam:PassedToService":"ec2.amazonaws.com"}}},
 {"Sid":"Logs","Effect":"Allow","Action":["s3:GetObject","s3:PutObject","s3:ListBucket"],
  "Resource":["arn:aws:s3:::noiz-t2s-us-east-2","arn:aws:s3:::noiz-t2s-us-east-2/*"]},
 {"Sid":"Wandb","Effect":"Allow","Action":"ssm:GetParameter",
  "Resource":"arn:aws:ssm:us-east-2:*:parameter/noiz-t2s/*"}]}
JSON
)

aws iam get-role --role-name "$ROLE" >/dev/null 2>&1 \
  || aws iam create-role --role-name "$ROLE" \
       --assume-role-policy-document "$TRUST" \
       --description "spot_supervisor.py host for the t2s-v1 run" >/dev/null
aws iam put-role-policy --role-name "$ROLE" \
  --policy-name NoizT2sSupervise --policy-document "$POLICY"
aws iam get-instance-profile --instance-profile-name "$PROFILE" >/dev/null 2>&1 || {
  aws iam create-instance-profile --instance-profile-name "$PROFILE" >/dev/null
  aws iam add-role-to-instance-profile --instance-profile-name "$PROFILE" \
    --role-name "$ROLE"
  # The profile is not usable the instant it is created.
  sleep 15
}
echo "role $ROLE, profile $PROFILE"

RUNNING=$(aws ec2 describe-instances --region "$REGION" --output text \
  --filters "Name=tag:Name,Values=$NAME" "Name=instance-state-name,Values=pending,running" \
  --query 'Reservations[].Instances[].InstanceId')
if [[ -n "$RUNNING" ]]; then
  echo "already up: $RUNNING"
else
  AMI=$(aws ssm get-parameter --region "$REGION" --output text --query Parameter.Value \
    --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64)
  ID=$(aws ec2 run-instances --region "$REGION" --output text \
    --query 'Instances[0].InstanceId' \
    --image-id "$AMI" --instance-type t3.small --subnet-id "$SUBNET" \
    --key-name "$KEY" --security-group-ids "$SG" --associate-public-ip-address \
    --iam-instance-profile "Name=$PROFILE" \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":30,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
    --user-data 'file:///dev/stdin' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME},{Key=Project,Value=t2s-v1}]" \
    --count 1 <<'USERDATA'
#!/bin/bash
set -x
dnf install -y -q python3-pip tmux tar
mkdir -p /home/ec2-user/t2s-repo
aws s3 cp s3://noiz-t2s-us-east-2/_staging/node-config.sh /home/ec2-user/node-config.sh --region us-east-2
TARBALL=$(aws s3 ls s3://noiz-t2s-us-east-2/_staging/ --region us-east-2 | awk '/t2s-repo-.*tar.gz/{print $4}' | tail -1)
aws s3 cp "s3://noiz-t2s-us-east-2/_staging/$TARBALL" /tmp/repo.tar.gz --region us-east-2
tar xzf /tmp/repo.tar.gz -C /home/ec2-user/t2s-repo --strip-components=1
chown -R ec2-user:ec2-user /home/ec2-user
USERDATA
)
  echo "launched $ID"
  aws ec2 wait instance-running --region "$REGION" --instance-ids "$ID"
fi

IP=$(aws ec2 describe-instances --region "$REGION" --output text \
  --filters "Name=tag:Name,Values=$NAME" "Name=instance-state-name,Values=running" \
  --query 'Reservations[].Instances[].PublicIpAddress')
echo "supervisor $IP"
echo "next: scp -i secrets/$KEY.pem secrets/$KEY.pem ec2-user@$IP:~/.ssh/  # nodes are private-only"
