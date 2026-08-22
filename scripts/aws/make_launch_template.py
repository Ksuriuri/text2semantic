#!/usr/bin/env python3
"""Create (or version) the EC2 launch template the p5 training nodes come from.

This is what the Spot supervisor recreates a preempted node from, so everything a
node needs that is *not* run-specific belongs here, and nothing that is. The split
is deliberate: run-specific values live in s3://BUCKET/_staging/node-config.sh,
which the user-data sources, so switching from the preflight to the real launch
does not need a new template version and a node recreated mid-run picks up the
current config by itself.

The part worth reading twice is the network block. p5.48xlarge has 32 network
cards at 100 Gbps each, and reaching that 3200 Gbps means one `efa` interface on
card 0 plus an `efa-only` interface on each of cards 1-31. EC2 will not
auto-assign a public IPv4 address to an instance with more than one interface, so
these nodes are private-only by construction -- hence the NAT gateway and the S3
gateway endpoint that setup_network.sh creates, and hence a supervisor that has to
run inside the VPC.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path

REGION = "us-east-2"
NAME = "t2s-p5-spot"
SUBNET = "subnet-0cb2d0676e616e018"          # noiz-t2s-p5-private, us-east-2a
SECURITY_GROUP = "sg-07840151fef77b3fe"      # noiz-t2s-efa: all traffic to itself
PLACEMENT_GROUP = "noiz-t2s-p5"
PROFILE = "NoizT2sStagingProfile"
KEY_NAME = "noiz-t2s-20260819"
# Ubuntu 24.04 because the supervisor's default ssh user is `ubuntu`, and the OSS
# PyTorch DLAMI because what is actually wanted from the image is the driver, the
# EFA installer and aws-ofi-nccl; the repo's own torch comes from `uv`.
AMI_PARAM = ("/aws/service/deeplearning/ami/x86_64/"
             "oss-nvidia-driver-gpu-pytorch-2.9-ubuntu-24.04/latest/ami-id")
CARDS = 32


def aws(*args):
    result = subprocess.run(["aws", *args, f"--region={REGION}"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"{' '.join(args)}\n{result.stderr.strip()}")
    return json.loads(result.stdout) if result.stdout.strip() else {}


def interfaces():
    """One efa on card 0, one efa-only on each remaining card.

    Card 0 is the only one that carries IP traffic, so it is the only one with a
    device index of 0; the rest exist purely for NCCL and take device index 1 on
    their own card. Every one of them still needs the subnet spelled out --
    RunInstances rejects the whole request with "Each network interface requires
    either a subnet or a network interface ID" otherwise, even for an efa-only
    interface that will never hold an address.
    """
    spec = [{
        "NetworkCardIndex": 0,
        "DeviceIndex": 0,
        "InterfaceType": "efa",
        "SubnetId": SUBNET,
        "Groups": [SECURITY_GROUP],
        "DeleteOnTermination": True,
    }]
    spec += [{
        "NetworkCardIndex": card,
        "DeviceIndex": 1,
        "InterfaceType": "efa-only",
        "SubnetId": SUBNET,
        "Groups": [SECURITY_GROUP],
        "DeleteOnTermination": True,
    } for card in range(1, CARDS)]
    return spec


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-data", default=str(Path(__file__).parent / "node_userdata.sh"))
    parser.add_argument("--name", default=NAME)
    parser.add_argument("--instance-type", default="p5.48xlarge")
    parser.add_argument("--root-gb", type=int, default=200)
    parser.add_argument("--on-demand", action="store_true",
                        help="omit the Spot market options, e.g. to hold a node "
                             "through a debug session without a preemption")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ami = aws("ssm", "get-parameter", f"--name={AMI_PARAM}",
              "--output=json")["Parameter"]["Value"]
    image = aws("ec2", "describe-images", f"--image-ids={ami}", "--output=json")["Images"][0]
    root = image["RootDeviceName"]
    print(f"ami {ami} ({image['Name']}), root {root}", file=sys.stderr)

    user_data = Path(args.user_data).read_bytes()
    data = {
        "ImageId": ami,
        "InstanceType": args.instance_type,
        "KeyName": KEY_NAME,
        "IamInstanceProfile": {"Name": PROFILE},
        "Placement": {"GroupName": PLACEMENT_GROUP, "AvailabilityZone": "us-east-2a"},
        "NetworkInterfaces": interfaces(),
        "BlockDeviceMappings": [{
            "DeviceName": root,
            # The dataset and the venv live on the striped instance store; this
            # only has to hold the image, the driver and the logs.
            "Ebs": {"VolumeSize": args.root_gb, "VolumeType": "gp3",
                    "DeleteOnTermination": True},
        }],
        "UserData": base64.b64encode(user_data).decode(),
        "MetadataOptions": {
            # The bootstrap names its own log in S3 after the Name tag, which is
            # only readable from IMDS when this is on.
            "InstanceMetadataTags": "enabled",
            "HttpTokens": "required",
        },
        "TagSpecifications": [{
            "ResourceType": "instance",
            "Tags": [{"Key": "Project", "Value": "t2s-v1"}],
        }],
    }
    if not args.on_demand:
        # In the template rather than on the command line: the supervisor's repair
        # path is a bare run-instances against this template, so a market option it
        # does not pass would silently turn a $40/h node into a $98/h one.
        data["InstanceMarketOptions"] = {
            "MarketType": "spot",
            "SpotOptions": {"SpotInstanceType": "one-time"},
        }

    if args.dry_run:
        print(json.dumps(data, indent=2))
        return 0

    existing = aws("ec2", "describe-launch-templates", "--output=json",
                   f"--filters=Name=launch-template-name,Values={args.name}")
    payload = json.dumps(data)
    if existing.get("LaunchTemplates"):
        result = aws("ec2", "create-launch-template-version", "--output=json",
                     f"--launch-template-name={args.name}",
                     f"--launch-template-data={payload}",
                     "--source-version=$Latest")
        version = result["LaunchTemplateVersion"]["VersionNumber"]
        aws("ec2", "modify-launch-template", "--output=json",
            f"--launch-template-name={args.name}", f"--default-version={version}")
        print(f"{args.name} version {version} (now default)")
    else:
        aws("ec2", "create-launch-template", "--output=json",
            f"--launch-template-name={args.name}",
            f"--launch-template-data={payload}",
            "--tag-specifications=ResourceType=launch-template,"
            "Tags=[{Key=Project,Value=t2s-v1}]")
        print(f"{args.name} version 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
