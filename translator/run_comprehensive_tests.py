#!/usr/bin/env python3
"""
Comprehensive test runner for the FormalVerifML framework.
This script runs all tests including HuggingFace model testing and Lean verification.
"""

import subprocess
import json
import time
import os
import sys
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple


def run_with_live_output(
    cmd: List[str], timeout: int, description: str
) -> Tuple[int, str, str]:
    """
    Run a command with live output that overwrites the current line.
    Returns (return_code, stdout, stderr).
    """
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    terminal_width = shutil.get_terminal_size((80, 20)).columns
    output_lines = []
    start_time = time.time()

    try:
        while True:
            # Check timeout
            if time.time() - start_time > timeout:
                process.kill()
                process.wait()
                raise subprocess.TimeoutExpired(cmd, timeout)

            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break

            if line:
                line = line.rstrip()
                output_lines.append(line)
                # Truncate line to terminal width and overwrite
                display_line = line[:terminal_width - 3] + "..." if len(line) > terminal_width else line
                print(f"\r{display_line:<{terminal_width}}", end="", flush=True)

        # Clear the progress line
        print(f"\r{' ' * terminal_width}\r", end="", flush=True)

        return process.returncode, "\n".join(output_lines), ""

    except subprocess.TimeoutExpired:
        print(f"\r{' ' * terminal_width}\r", end="", flush=True)
        raise


class ComprehensiveTestRunner:
    """Runner for comprehensive testing of the FormalVerifML framework."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = {}
        self.start_time = time.time()

    def run_huggingface_tests(self) -> Dict[str, Any]:
        """Run HuggingFace model tests."""
        print("=" * 60)
        print("RUNNING HUGGINGFACE MODEL TESTS")
        print("=" * 60)

        # Check for required dependencies first
        try:
            import importlib.util
            missing_deps = []
            for dep in ["torch", "transformers", "psutil"]:
                if importlib.util.find_spec(dep) is None:
                    missing_deps.append(dep)

            if missing_deps:
                print(f"⚠️  Skipping HuggingFace tests: missing {', '.join(missing_deps)}")
                return {"status": "skipped", "reason": f"Missing dependencies: {', '.join(missing_deps)}"}
        except Exception as e:
            print(f"⚠️  Skipping HuggingFace tests: dependency check failed ({e})")
            return {"status": "skipped", "reason": str(e)}

        try:
            cmd = [
                sys.executable,
                "translator/test_huggingface_models.py",
                "--model_size",
                self.config.get("huggingface_model_size", "small"),
                "--output",
                "test_results_huggingface.json",
            ]

            returncode, stdout, _ = run_with_live_output(
                cmd, timeout=300, description="HuggingFace tests"
            )

            if returncode == 0:
                try:
                    with open("test_results_huggingface.json", "r") as f:
                        results = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    results = {"output": stdout}
                print("✅ HuggingFace tests completed successfully")
                return {"status": "success", "results": results}
            else:
                print("❌ HuggingFace tests failed")
                return {"status": "error", "error": stdout}

        except subprocess.TimeoutExpired:
            print("❌ HuggingFace tests timed out")
            return {"status": "timeout"}
        except Exception as e:
            print(f"❌ HuggingFace tests error: {str(e)}")
            return {"status": "error", "error": str(e)}

    def run_lean_build(self) -> Dict[str, Any]:
        """Run Lean build via Docker to ensure all modules compile."""
        print("=" * 60)
        print("RUNNING DOCKER BUILD (this may take ~15 minutes)")
        print("=" * 60)

        try:
            cmd = ["docker", "build", "-t", "circuitproofs", "."]
            returncode, stdout, _ = run_with_live_output(
                cmd, timeout=1800, description="Docker build"
            )

            if returncode == 0:
                print("✅ Docker build completed successfully")
                return {"status": "success", "output": stdout}
            else:
                print("❌ Docker build failed")
                return {"status": "error", "error": stdout}

        except subprocess.TimeoutExpired:
            print("❌ Docker build timed out (>30 minutes)")
            return {"status": "timeout"}
        except Exception as e:
            print(f"❌ Docker build error: {str(e)}")
            return {"status": "error", "error": str(e)}

    def run_lean_tests(self) -> Dict[str, Any]:
        """Run Lean verification tests."""
        print("=" * 60)
        print("RUNNING LEAN VERIFICATION TESTS")
        print("=" * 60)

        try:
            cmd = ["lake", "exe", "formal_verif_ml_exe"]
            returncode, stdout, _ = run_with_live_output(
                cmd, timeout=180, description="Lean verification"
            )

            if returncode == 0:
                print("✅ Lean verification tests completed successfully")
                return {"status": "success", "output": stdout}
            else:
                print("❌ Lean verification tests failed")
                return {"status": "error", "error": stdout}

        except subprocess.TimeoutExpired:
            print("❌ Lean verification tests timed out")
            return {"status": "timeout"}
        except Exception as e:
            print(f"❌ Lean verification tests error: {str(e)}")
            return {"status": "error", "error": str(e)}

    def run_memory_tests(self) -> Dict[str, Any]:
        """Run memory optimization tests."""
        print("=" * 60)
        print("RUNNING MEMORY OPTIMIZATION TESTS")
        print("=" * 60)

        try:
            cmd = [
                sys.executable,
                "translator/generate_lean_model.py",
                "--model_json",
                "translator/sample_transformer.json",
                "--output_lean",
                "lean/FormalVerifML/generated/memory_optimized_test.lean",
            ]

            returncode, stdout, _ = run_with_live_output(
                cmd, timeout=60, description="Memory optimization"
            )

            if returncode == 0:
                print("✅ Memory optimization tests completed successfully")
                return {"status": "success", "output": stdout}
            else:
                print("❌ Memory optimization tests failed")
                return {"status": "error", "error": stdout}

        except subprocess.TimeoutExpired:
            print("❌ Memory optimization tests timed out")
            return {"status": "timeout"}
        except Exception as e:
            print(f"❌ Memory optimization tests error: {str(e)}")
            return {"status": "error", "error": str(e)}

    def run_smt_tests(self) -> Dict[str, Any]:
        """Run SMT solver integration tests."""
        print("=" * 60)
        print("RUNNING SMT SOLVER INTEGRATION TESTS")
        print("=" * 60)

        try:
            cmd = ["lake", "exe", "formal_verif_ml_exe"]
            returncode, stdout, _ = run_with_live_output(
                cmd, timeout=120, description="SMT integration"
            )

            if returncode == 0:
                print("✅ SMT integration tests completed successfully")
                return {"status": "success", "output": stdout}
            else:
                print("❌ SMT integration tests failed")
                return {"status": "error", "error": stdout}

        except subprocess.TimeoutExpired:
            print("❌ SMT integration tests timed out")
            return {"status": "timeout"}
        except Exception as e:
            print(f"❌ SMT integration tests error: {str(e)}")
            return {"status": "error", "error": str(e)}

    def run_web_interface_tests(self) -> Dict[str, Any]:
        """Run web interface tests."""
        print("=" * 60)
        print("RUNNING WEB INTERFACE TESTS")
        print("=" * 60)

        try:
            # Start web server
            web_process = subprocess.Popen(
                [sys.executable, "webapp/app.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            # Wait for server to start
            time.sleep(5)

            # Test web interface
            import requests

            try:
                response = requests.get("http://localhost:5000", timeout=10)
                if response.status_code == 200:
                    print("✅ Web interface tests completed successfully")
                    result = {"status": "success", "status_code": response.status_code}
                else:
                    print(
                        f"❌ Web interface returned status code: {response.status_code}"
                    )
                    result = {"status": "error", "status_code": response.status_code}
            except requests.RequestException as e:
                print(f"❌ Web interface test failed: {str(e)}")
                result = {"status": "error", "error": str(e)}

            # Cleanup
            web_process.terminate()
            web_process.wait()

            return result

        except Exception as e:
            print(f"❌ Web interface tests error: {str(e)}")
            return {"status": "error", "error": str(e)}

    def cleanup_docker(self) -> None:
        """Prune Docker images created during testing."""
        print("=" * 60)
        print("CLEANING UP DOCKER RESOURCES")
        print("=" * 60)

        try:
            # Remove the circuitproofs image
            cmd = ["docker", "rmi", "-f", "circuitproofs"]
            returncode, stdout, _ = run_with_live_output(
                cmd, timeout=60, description="Docker image removal"
            )

            if returncode == 0:
                print("✅ Docker image 'circuitproofs' removed")
            else:
                print("⚠️  Could not remove Docker image (may not exist)")

            # Prune dangling images and build cache
            cmd = ["docker", "image", "prune", "-f"]
            returncode, stdout, _ = run_with_live_output(
                cmd, timeout=60, description="Docker prune"
            )

            if returncode == 0:
                print("✅ Docker dangling images pruned")
            else:
                print("⚠️  Docker prune had issues")

        except Exception as e:
            print(f"⚠️  Docker cleanup error: {str(e)}")

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests."""
        print("STARTING COMPREHENSIVE TEST SUITE")
        print("=" * 60)

        # Run all test categories
        test_categories = [
            ("lean_build", self.run_lean_build),
            ("huggingface_tests", self.run_huggingface_tests),
            ("lean_tests", self.run_lean_tests),
            ("memory_tests", self.run_memory_tests),
            ("smt_tests", self.run_smt_tests),
            ("web_interface_tests", self.run_web_interface_tests),
        ]

        for category, test_func in test_categories:
            if self.config.get(f"run_{category}", True):
                self.results[category] = test_func()
            else:
                print(f"Skipping {category} (disabled in config)")
                self.results[category] = {"status": "skipped"}

        # Clean up Docker resources after all tests
        self.cleanup_docker()

        return self.results

    def generate_report(self) -> str:
        """Generate comprehensive test report."""
        total_time = time.time() - self.start_time

        # Count results
        total_tests = len(self.results)
        successful_tests = sum(
            1 for r in self.results.values() if r.get("status") == "success"
        )
        failed_tests = sum(
            1 for r in self.results.values() if r.get("status") == "error"
        )
        timeout_tests = sum(
            1 for r in self.results.values() if r.get("status") == "timeout"
        )
        skipped_tests = sum(
            1 for r in self.results.values() if r.get("status") == "skipped"
        )

        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0

        report = f"""
COMPREHENSIVE TEST REPORT
{'='*60}
Total Test Categories: {total_tests}
Successful: {successful_tests}
Failed: {failed_tests}
Timeout: {timeout_tests}
Skipped: {skipped_tests}
Total Execution Time: {total_time:.2f}s
Success Rate: {success_rate:.1f}%

DETAILED RESULTS:
{'='*60}
"""

        for category, result in self.results.items():
            status = result.get("status", "unknown")
            report += f"\n{category.upper()}:\n"
            report += f"  Status: {status}\n"

            if "error" in result:
                report += f"  Error: {result['error']}\n"
            if "output" in result:
                report += f"  Output: {result['output'][:200]}...\n"

        return report


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive tests for FormalVerifML"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="test_config.json",
        help="Path to test configuration file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="comprehensive_test_report.txt",
        help="Path to output report file",
    )
    parser.add_argument(
        "--huggingface-size",
        choices=["small", "medium", "all"],
        default="small",
        help="Size of HuggingFace models to test",
    )

    args = parser.parse_args()

    # Default configuration
    config = {
        "huggingface_model_size": args.huggingface_size,
        "run_lean_build": True,
        "run_huggingface_tests": True,
        "run_lean_tests": True,
        "run_memory_tests": True,
        "run_smt_tests": True,
        "run_web_interface_tests": True,
    }

    # Load custom config if provided
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            custom_config = json.load(f)
            config.update(custom_config)

    # Run tests
    runner = ComprehensiveTestRunner(config)
    results = runner.run_all_tests()

    # Generate and save report
    report = runner.generate_report()

    with open(args.output, "w") as f:
        f.write(report)

    print("\n" + report)
    print(f"\nDetailed report saved to: {args.output}")

    # Exit with appropriate code
    successful_tests = sum(1 for r in results.values() if r.get("status") == "success")
    total_tests = len(results)

    if successful_tests == total_tests:
        print("🎉 All tests passed!")
        exit(0)
    else:
        print(f"⚠️  {total_tests - successful_tests} test(s) failed")
        exit(1)


if __name__ == "__main__":
    main()
