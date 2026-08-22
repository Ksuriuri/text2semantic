#!/bin/bash
# Network for the p5 training nodes. Idempotent: re-running it finds what exists.
#
# Why any of this is needed. p5.48xlarge reaches its 3200 Gbps with 32 EFA
# interfaces, and EC2 refuses to auto-assign a public IPv4 address to an instance
# with more than one network interface. So a full-width p5 has no public address,
# and this VPC has neither a NAT gateway nor a VPC endpoint -- a node would come up
# with no route to PyPI or to W&B, and W&B logging is a requirement of the run.
#
# Two things follow, and the second one is the expensive one to get wrong:
#
#   * a private subnet with its own route table, so the shared default VPC's main
#     route table is left exactly as other teams' instances found it;
#   * an S3 *gateway* endpoint on that route table. Without it the 8.4 TB each node
#     pulls would be billed as NAT data processing: 8 nodes x 8.4 TB x $0.045/GB is
#     about $3,000, against $0 through the endpoint.
#
# The NAT gateway itself then carries only PyPI wheels and W&B traffic, a few
# dollars. It is a standing resource: delete it when the run is done.
set -euo pipefail

REGION=us-east-2
VPC=vpc-0634c032c5b5e7a7a
PUBLIC_SUBNET=subnet-0d364f8ca36b5d85b   # us-east-2a, routes to the internet gateway
AZ=us-east-2a
CIDR=172.31.64.0/20                       # the /16 has 0/20, 16/20 and 32/20 taken
NAME=noiz-t2s-p5
PG=noiz-t2s-p5

q() { aws ec2 "$@" --region "$REGION" --output text; }

# ----------------------------------------------------------------- private subnet
SUBNET=$(q describe-subnets --filters "Name=vpc-id,Values=$VPC" \
  "Name=tag:Name,Values=$NAME-private" --query 'Subnets[0].SubnetId')
if [[ "$SUBNET" == "None" || -z "$SUBNET" ]]; then
  SUBNET=$(q create-subnet --vpc-id "$VPC" --cidr-block "$CIDR" \
    --availability-zone "$AZ" \
    --tag-specifications "ResourceType=subnet,Tags=[{Key=Name,Value=$NAME-private},{Key=Project,Value=t2s-v1}]" \
    --query 'Subnet.SubnetId')
  echo "created subnet $SUBNET"
fi
echo "subnet $SUBNET"

# -------------------------------------------------------------------- nat gateway
NAT=$(q describe-nat-gateways --filter "Name=vpc-id,Values=$VPC" \
  "Name=tag:Name,Values=$NAME-nat" "Name=state,Values=pending,available" \
  --query 'NatGateways[0].NatGatewayId')
if [[ "$NAT" == "None" || -z "$NAT" ]]; then
  EIP=$(q allocate-address --domain vpc \
    --tag-specifications "ResourceType=elastic-ip,Tags=[{Key=Name,Value=$NAME-nat}]" \
    --query 'AllocationId')
  # In the *public* subnet on purpose: a NAT gateway is only useful where it can
  # itself reach the internet gateway.
  NAT=$(q create-nat-gateway --subnet-id "$PUBLIC_SUBNET" --allocation-id "$EIP" \
    --tag-specifications "ResourceType=natgateway,Tags=[{Key=Name,Value=$NAME-nat},{Key=Project,Value=t2s-v1}]" \
    --query 'NatGateway.NatGatewayId')
  echo "created nat $NAT, waiting"
  aws ec2 wait nat-gateway-available --region "$REGION" --nat-gateway-ids "$NAT"
fi
echo "nat $NAT"

# --------------------------------------------------------------------- route table
RTB=$(q describe-route-tables --filters "Name=vpc-id,Values=$VPC" \
  "Name=tag:Name,Values=$NAME-private" --query 'RouteTables[0].RouteTableId')
if [[ "$RTB" == "None" || -z "$RTB" ]]; then
  RTB=$(q create-route-table --vpc-id "$VPC" \
    --tag-specifications "ResourceType=route-table,Tags=[{Key=Name,Value=$NAME-private},{Key=Project,Value=t2s-v1}]" \
    --query 'RouteTable.RouteTableId')
  echo "created route table $RTB"
fi
aws ec2 create-route --region "$REGION" --route-table-id "$RTB" \
  --destination-cidr-block 0.0.0.0/0 --nat-gateway-id "$NAT" >/dev/null 2>&1 \
  || aws ec2 replace-route --region "$REGION" --route-table-id "$RTB" \
       --destination-cidr-block 0.0.0.0/0 --nat-gateway-id "$NAT"
ASSOC=$(q describe-route-tables --route-table-ids "$RTB" \
  --query "RouteTables[0].Associations[?SubnetId=='$SUBNET'].RouteTableAssociationId")
if [[ -z "$ASSOC" || "$ASSOC" == "None" ]]; then
  q associate-route-table --route-table-id "$RTB" --subnet-id "$SUBNET" >/dev/null
  echo "associated $RTB with $SUBNET"
fi
echo "route table $RTB"

# ------------------------------------------------------------------- s3 endpoint
# The whole reason the data pull is affordable. Gateway endpoints are free and are
# attached to route tables, not to instances.
ENDPOINT=$(q describe-vpc-endpoints --filters "Name=vpc-id,Values=$VPC" \
  "Name=service-name,Values=com.amazonaws.$REGION.s3" \
  --query "VpcEndpoints[?VpcEndpointType=='Gateway'].VpcEndpointId | [0]")
if [[ "$ENDPOINT" == "None" || -z "$ENDPOINT" ]]; then
  ENDPOINT=$(q create-vpc-endpoint --vpc-id "$VPC" \
    --service-name "com.amazonaws.$REGION.s3" --vpc-endpoint-type Gateway \
    --route-table-ids "$RTB" \
    --tag-specifications "ResourceType=vpc-endpoint,Tags=[{Key=Name,Value=$NAME-s3},{Key=Project,Value=t2s-v1}]" \
    --query 'VpcEndpoint.VpcEndpointId')
  echo "created s3 endpoint $ENDPOINT"
else
  # Only ever adds this run's route table; another team's tables stay untouched.
  aws ec2 modify-vpc-endpoint --region "$REGION" --vpc-endpoint-id "$ENDPOINT" \
    --add-route-table-ids "$RTB" >/dev/null 2>&1 || true
fi
echo "s3 endpoint $ENDPOINT"

# --------------------------------------------------------------- placement group
if ! q describe-placement-groups --group-names "$PG" >/dev/null 2>&1; then
  q create-placement-group --group-name "$PG" --strategy cluster \
    --tag-specifications "ResourceType=placement-group,Tags=[{Key=Name,Value=$PG},{Key=Project,Value=t2s-v1}]" >/dev/null
  echo "created placement group $PG"
fi
echo "placement group $PG"

cat <<SUMMARY

subnet          $SUBNET   ($CIDR, $AZ, no public IPs)
nat             $NAT
route table     $RTB      (0.0.0.0/0 -> nat, s3 via endpoint)
s3 endpoint     $ENDPOINT
placement group $PG       (cluster)
SUMMARY
