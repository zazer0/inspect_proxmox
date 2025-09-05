import logging
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match
from inspect_ai.solver import basic_agent
from inspect_ai.tool import bash
from inspect_ai.util import SandboxEnvironmentSpec

from proxmoxsandbox._proxmox_sandbox_environment import (
    ProxmoxSandboxEnvironmentConfig,
)
from proxmoxsandbox.schema import VmConfig, VmSourceConfig

# Set up logging to see snapshot operations
logger = logging.getLogger(__name__)


@task
def test_snapshot() -> Task:
    """Test task to validate snapshot-based optimization for Ubuntu VMs.
    
    This test verifies that:
    1. On first run, a "post-cloudinit" snapshot is created after VM initialization
    2. On subsequent runs, the snapshot is detected and used for faster cloning
    
    Watch the logs for:
    - "Creating snapshot 'post-cloudinit'..." (first run)
    - "Snapshot 'post-cloudinit' already exists..." (subsequent runs)
    - "Found 'post-cloudinit' snapshot for VM..." (when cloning from snapshot)
    """
    
    return Task(
        dataset=[
            Sample(
                input="""Verify that this Ubuntu 24.04 system is working correctly.
                
Please run the following checks:
1. Check the OS version with 'cat /etc/os-release | grep VERSION='
2. Check the hostname with 'hostname'
3. Verify qemu-guest-agent is running with 'systemctl is-active qemu-guest-agent'
4. List network interfaces with 'ip -brief addr'

If all checks pass successfully, respond with exactly: "SNAPSHOT_TEST_SUCCESS"
""",
                target="SNAPSHOT_TEST_SUCCESS",
            )
        ],
        solver=[
            basic_agent(
                tools=[bash(timeout=60)],
                message_limit=10,
            ),
        ],
        scorer=match(),
        sandbox=SandboxEnvironmentSpec(
            type="proxmox",
            config=ProxmoxSandboxEnvironmentConfig(
                vms_config=(
                    VmConfig(
                        vm_source_config=VmSourceConfig(
                            # Using built_in triggers the snapshot optimization logic
                            built_in="ubuntu24.04"
                        ),
                        name="test-snapshot-vm",
                    ),
                ),
            ),
        ),
    )


if __name__ == "__main__":
    import asyncio
    from inspect_ai import eval
    
    # Configure logging to see snapshot operations
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("SNAPSHOT OPTIMIZATION TEST")
    print("=" * 60)
    print("\nThis test validates the snapshot-based VM optimization.")
    print("\nOn FIRST RUN, watch for:")
    print("  - VM creation from template")
    print("  - 'Creating snapshot post-cloudinit...' message")
    print("  - Longer initialization time (3-5 minutes)")
    print("\nOn SUBSEQUENT RUNS, watch for:")
    print("  - 'Found post-cloudinit snapshot for VM...' message")
    print("  - Faster initialization time (1-2 minutes)")
    print("=" * 60)
    print()
    
    # Run the evaluation
    result = asyncio.run(eval(
        test_snapshot(),
        model="openai/gpt-4o-mini",  # Use a simple model for testing
    ))
    
    # Report results
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    if result.results and result.results[0].score:
        print(f"✓ Test passed! Score: {result.results[0].score}")
        print("\nCheck the logs above to verify snapshot behavior:")
        print("  - First run should show snapshot creation")
        print("  - Subsequent runs should show snapshot reuse")
    else:
        print("✗ Test failed. Check the logs for details.")
    print("=" * 60)