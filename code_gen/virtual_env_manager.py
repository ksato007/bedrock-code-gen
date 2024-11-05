from pathlib import Path
import os
import sys
import subprocess
import shutil
import shlex
import json
import tempfile
from .utils import FileHandler
from .graph_state import SharedState
from .config_manager import ConfigManager
from .logger import setup_logging, log
import time
import stat

# set up logging
setup_logging('INFO')

class ProjectVirtualEnvironmentManager:
    """
    A class responsible for managing the project's virtual environment and test configuration.

    Attributes:
        state (SharedState): The shared state object containing the project's state.
        env_settings (dict): The environment settings loaded from the configuration file.
        venv_name (str): The name of the virtual environment.
        python_version (str): The version of Python to use for the virtual environment.
        project_name (str): The name of the project.

    Methods:
        create_venv(): Creates a virtual environment and installs the required dependencies.
        create_test_config(): Creates the necessary configuration files for running tests.
    """
    def __init__(self, state: SharedState):
        self.state = state
        config = ConfigManager.from_yaml_file(self.state.config_path)
        self.env_settings = config.env_settings
        self.venv_name = self.env_settings.get('venv_name', 'venv')
        self.python_version = self.env_settings.get('python_version', '3.10.14')
        self.project_name = self.env_settings.get('project_name', 'my_project')
        self.prompt_settings = config.prompt_settings

    def create_venv(self) -> SharedState:
        """
        Creates a virtual environment and installs the required dependencies.

        Returns:
            SharedState: The updated shared state with the virtual environment created.
        """
        log("Creating virtual environment", "INFO")
        # Create a virtual environment
        venv_name = self.venv_name
        project_name = self.project_name
        python_version = self.python_version
        # requirements = self.state.source_file_states['requirements'].content.strip()
        requirements = self.state.source_file_states['requirements'].solution['code'].strip()
        replacement_dict = {
            "\n": " ",
            "==": ">="
        }
        for old, new in replacement_dict.items():
            requirements = requirements.replace(old, new)
        env_manager = VirtualEnvironmentManager(
            project_name=project_name,
            venv_name=venv_name, 
            python_version=python_version
        )
        log("Installing dependencies", "INFO")
        env_manager.install_dependency(
            package_name= requirements
        )
        self.state.venv_created = True
        return self.state
    
    def check_venv_setup(self) -> bool:
        """
        Checks if the virtual environment is set up correctly.

        Returns:
            bool: True if the virtual environment is set up correctly, False otherwise.
        """
        log("Checking virtual environment setup", "INFO")
        cwd = Path.cwd()
        venv_dir = cwd / self.venv_name

        if venv_dir.exists() and venv_dir.is_dir():
            return True
        else:
            log(f"Virtual environment directory '{venv_dir}' does not exist or is not a directory.", "WARNING")
            return False

    def create_test_config(self, config_path: str = '') -> bool:
        """
        Creates the necessary configuration files for running tests.
        Returns:
            SharedState: The updated shared state.
        """
        venv_name = self.venv_name
        log("Creating pytest ini file", "INFO")
        FileHandler.create_pytest_ini(venv_name=venv_name, config_path=config_path)
        log("Creating tests/conftest.py file", "INFO")
        FileHandler.create_conftest_py(venv_name=venv_name, config_path=config_path)
        log("Creating .coveragerc file", "INFO")
        FileHandler.create_coveragerc(config_path=config_path)
        return True
    
class VirtualEnvironmentManager:
    """
    A class for managing virtual environments.
    To-dos:
        - currently it uses wrappers for python, pip, pytest to us especific python installed via pyenv. 
          this approach requires usage of wrapper scripts which can cause confusion and mismatch 
          this was a workaround for tensorflow install which does not support python 3.12 which is set as default in my PC
        - address below
            now: os.chmod(pip_wrapper_path, 0o755)  # Make the wrapper executable
            new: import stat
            # Make the wrapper executable with read, write, and execute permissions for the owner
            os.chmod(pip_wrapper_path, stat.S_IRWXU)
    """
    def __init__(self, project_name='', base_dir='', venv_name='',python_version=None):
        # Reset current directory to path to __file__ or Jupyter notebook
        self.is_windows = os.name == 'nt'
        self.reset_current_directory()
        self.base_dir = base_dir or os.getcwd()
        self.project_name = project_name or 'project_name'
        self.venv_name = venv_name or 'venv'
        self.python_path = VirtualEnvironmentManager.find_python(python_version)
        self.project_dir = self.create_and_change_directory()
        self.venv_path = os.path.join(self.project_dir, venv_name)
        self.create_venv_with_specific_python()

    def delete_venv(self):
        """
        Deletes a virtual environment and all its contents.
        """
        venv_path = Path(self.venv_path)
        if venv_path.xexists():
            subprocess.run([sys.executable, "-m", "venv", "--clear", venv_path], check=True)
            shutil.rmtree(venv_path, ignore_errors=True)
            log(f"Virtual environment '{venv_path}' has been deleted.", "INFO")

    @staticmethod
    def find_python(version):
        """Find the path to the specified Python version, considering pyenv."""
        
        # Check if pyenv is installed and in use
        try:
            pyenv_root = subprocess.check_output(["pyenv", "root"], universal_newlines=True).strip()
            pyenv_version_path = os.path.join(pyenv_root, "versions", version, "bin", "python")
            if os.path.exists(pyenv_version_path):
                return pyenv_version_path
        except subprocess.CalledProcessError:
            # pyenv not found or not in use
            pass

        # Standard locations to check
        potential_paths = [
            f"/usr/bin/python{version}",
            f"/usr/local/bin/python{version}",
            f"C:\\Python{version.replace('.', '')}\\python.exe",  # Windows
            str(Path.home() / f"AppData/Local/Programs/Python/Python{version.replace('.', '')}/python.exe"),  # Windows user install
            f"/opt/homebrew/bin/python{version}",  # Homebrew on macOS
            f"/usr/local/opt/python{version}/bin/python{version}", #macOS with Homebrew
            f"/Library/Frameworks/Python.framework/Versions/{version}/bin/python{version}",  # macOS framework
            f"/opt/python{version}/bin/python{version}", #some Linux distributions use this path
            f"%LOCALAPPDATA%\\Programs\\Python\\Python{version}\\python.exe",  # Windows user install (alternative path)
        ]
        
        # Add paths from PATH environment variable
        path_dirs = os.environ.get('PATH', '').split(os.pathsep)
        for path_dir in path_dirs:
            potential_paths.append(str(Path(path_dir) / f"python{version}"))
            potential_paths.append(str(Path(path_dir) / "python"))
        
        for path in potential_paths:
            if os.path.exists(path):
                try:
                    result = subprocess.run([path, "--version"], capture_output=True, text=True)
                    if version in result.stdout:
                        return path
                except:
                    continue
        raise ValueError(f"Python {version} not found. Make sure it's installed and accessible.")
    
    def reset_current_directory(self):
        """
        Resets the current working directory to the directory where the script/notebook is located.
        """
        try:
            # Try to get the directory of the current script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(script_dir)
            script_dir = parent_dir
            log(f"Script path: {script_dir}", "INFO")
        except (NameError, AttributeError):
            # If __file__ is not defined, get the directory of the current Jupyter Notebook
            from jupyter_server import serverapp
            notebook_path = serverapp.ServerApp().root_dir
            log(f"Notebook path: {notebook_path}", "INFO")
            script_dir = str(notebook_path)

        os.chdir(script_dir)
        log(f"Current working directory set to: {os.getcwd()}", "INFO")

    def create_and_change_directory(self):
        """
        Changes the current working directory to the directory corresponding to the given project name.

        Args:
            project_name (str): The name of the project for which the directory should be changed.

        Raises:
            ValueError: If the project name is not found in the expected directory structure.
        """
        # if project name folder exists in base_dir, do nothing and return base_dir
        if self.project_name in self.base_dir:
            return self.base_dir
        # Define the expected directory structure
        project_dir = os.path.join(self.base_dir, self.project_name)
        # Check if the project directory exists
        if os.path.isdir(project_dir):
            # Change the current working directory to the project directory
            os.chdir(project_dir)
            log(f"Changed directory to: {project_dir}", "INFO")
        else:
            log(f"Project directory '{project_dir}'", "INFO")
            os.makedirs(project_dir)
            log(f"Created directory: {project_dir}", "INFO")
            os.chdir(project_dir)
            log(f"Changed directory to: {project_dir}", "INFO")
        return project_dir

    def create_venv_with_specific_python(self):
        # Create the virtual environment using the specified Python interpreter
        try:
            subprocess.run([self.python_path, "-m", "venv", self.venv_path], check=True)
        except subprocess.CalledProcessError as e:
            log(f"Error creating virtual environment: {e}", "ERROR")
            return

        # Create a wrapper script to ensure the correct Python interpreter is used
        if os.name == 'nt':  # Windows
            # Create a batch file wrapper script in the Scripts directory
            wrapper_path = os.path.join(self.venv_path, "Scripts", "python-wrapper.bat")
            with open(wrapper_path, "w") as f:
                f.write(f'@echo off\n"{self.python_path}" %*')
            
            # Update the activate.bat file to use the wrapper script
            activate_path = os.path.join(self.venv_path, "Scripts", "activate.bat")
            with open(activate_path, "r+") as f:
                content = f.read()
                f.seek(0, 0)
                f.write(f'set "PYTHONPATH={os.path.dirname(wrapper_path)}"\n')
                f.write(f'set "PATH={os.path.dirname(wrapper_path)};%PATH%"\n')
                f.write(content)

            # Create a pip wrapper
            pip_wrapper_path = os.path.join(self.venv_path, "Scripts", "pip-wrapper.bat")
            with open(pip_wrapper_path, "w") as f:
                f.write(f'@echo off\n"{self.python_path}" -m pip %*')

            # Replace pip.exe with the wrapper
            os.rename(os.path.join(self.venv_path, "Scripts", "pip.exe"),
                      os.path.join(self.venv_path, "Scripts", "pip.exe.bak"))
            os.rename(pip_wrapper_path, os.path.join(self.venv_path, "Scripts", "pip.exe"))

            # Create a pytest wrapper
            pytest_wrapper_path = os.path.join(self.venv_path, "Scripts", "pytest-wrapper.bat")
            with open(pytest_wrapper_path, "w") as f:
                f.write(f'@echo off\n"{self.python_path}" -m pytest %*')

            # Create or replace pytest.exe with the wrapper
            pytest_path = os.path.join(self.venv_path, "Scripts", "pytest.exe")
            if os.path.exists(pytest_path):
                os.rename(pytest_path, f"{pytest_path}.bak")
            os.rename(pytest_wrapper_path, pytest_path)
        else:  # Unix-like systems
            # Create a shell script wrapper in the bin directory
            wrapper_path = os.path.join(self.venv_path, "bin", "python-wrapper")
            with open(wrapper_path, "w") as f:
                f.write(f'#!/bin/sh\nexec "{shlex.quote(self.python_path)}" "$@"')
            os.chmod(wrapper_path, stat.S_IRWXU)  # Make the wrapper executable for the owner only
            
            # Update the activate script to use the wrapper script
            activate_path = os.path.join(self.venv_path, "bin", "activate")
            with open(activate_path, "r+") as f:
                content = f.read()
                f.seek(0, 0)
                f.write(f'export PYTHONPATH="{os.path.dirname(wrapper_path)}"\n')
                f.write(f'export PATH="{os.path.dirname(wrapper_path)}:$PATH"\n')
                f.write(content)

            # Create a new python symlink to the wrapper script
            os.remove(os.path.join(self.venv_path, "bin", "python"))
            os.symlink(wrapper_path, os.path.join(self.venv_path, "bin", "python"))

            # Create a pip wrapper
            pip_wrapper_path = os.path.join(self.venv_path, "bin", "pip-wrapper")
            with open(pip_wrapper_path, "w") as f:
                f.write(f'#!/bin/sh\nexec "{shlex.quote(self.python_path)}" -m pip "$@"')
            os.chmod(pip_wrapper_path, stat.S_IRWXU)  # Make the wrapper executable for the owner only


            # Replace pip with the wrapper
            os.remove(os.path.join(self.venv_path, "bin", "pip"))
            os.symlink(pip_wrapper_path, os.path.join(self.venv_path, "bin", "pip"))

           # Create a pytest wrapper
            pytest_wrapper_path = os.path.join(self.venv_path, "bin", "pytest-wrapper")
            with open(pytest_wrapper_path, "w") as f:
                f.write(f'#!/bin/sh\nexec "{shlex.quote(self.python_path)}" -m pytest "$@"')
            os.chmod(pytest_wrapper_path, stat.S_IRWXU)  # Make the wrapper executable for the owner only

            # Create or replace pytest symlink with the wrapper
            pytest_path = os.path.join(self.venv_path, "bin", "pytest")
            if os.path.exists(pytest_path):
                os.remove(pytest_path)
            os.symlink(pytest_wrapper_path, pytest_path)

        # Print the virtual environment creation details and the activation command
        log(f"Virtual environment created at {self.venv_path} using Python from {self.python_path}", "INFO")
        if os.name == 'nt':  # Windows
            log(f"To activate, run: {self.venv_path}\\Scripts\\activate.bat", "INFO")
        else:  # Unix-like systems
            log(f"To activate, run: source {self.venv_path}/bin/activate", "INFO")

        # Verify the Python version and path after creating the virtual environment
        time.sleep(5)  # Wait for 5 seconds
        self.verify_python_version()
        self.verify_python_path()
        self.verify_pip_version()
        self.verify_pytest_version()

    def verify_python_version(self):
        try:
            if os.name == 'nt':  # Windows
                venv_python = os.path.join(self.venv_path, "Scripts", "python.exe")
            else:  # Unix-like systems
                venv_python = os.path.join(self.venv_path, "bin", "python")
            
            result = subprocess.run([venv_python, "--version"], capture_output=True, text=True, check=True)
            if result.returncode == 0:
                log(f"Python version in the virtual environment: {result.stdout.strip()}", "INFO")
            else:
                log(f"Error checking Python version: {result.stderr.strip()}", "INFO")
        except subprocess.CalledProcessError as e:
            log(f"Error checking Python version: {e.stderr.strip()}", "ERROR")
        except Exception as e:
            log(f"Error executing Python in the virtual environment: {e}", "ERROR")
            raise

    def verify_python_path(self):
        try:
            if os.name == 'nt':  # Windows
                venv_python = os.path.join(self.venv_path, "Scripts", "python.exe")
            else:  # Unix-like systems
                venv_python = os.path.join(self.venv_path, "bin", "python")
            
            result = subprocess.run([venv_python, "-c", "import sys; print(sys.executable)"], capture_output=True, text=True, check=True)
            if result.returncode == 0:
                log(f"Python path in the virtual environment: {result.stdout.strip()}", "INFO")
            else:
                log(f"Error checking Python path: {result.stderr.strip()}", "ERROR")
        except subprocess.CalledProcessError as e:
            log(f"Error checking Python path: {e.stderr.strip()}", "ERROR")
        except Exception as e:
            log(f"Error executing Python in the virtual environment: {e}", "ERROR")
            raise

    def verify_pip_version(self):
        try:
            if os.name == 'nt':  # Windows
                venv_pip = os.path.join(self.venv_path, "Scripts", "pip.exe")
            else:  # Unix-like systems
                venv_pip = os.path.join(self.venv_path, "bin", "pip")
            
            result = subprocess.run([venv_pip, "--version"], capture_output=True, text=True, check=True)
            if result.returncode == 0:
                log(f"Pip version in the virtual environment: {result.stdout.strip()}", "INFO")
            else:
                log(f"Error checking Pip version: {result.stderr.strip()}", "ERROR")
        except subprocess.CalledProcessError as e:
            log(f"Error checking Pip version: {e.stderr.strip()}", "ERROR")
        except Exception as e:
            log(f"Error executing Pip in the virtual environment: {e}", "ERROR")
            raise

    def verify_pytest_version(self):
        try:
            if os.name == 'nt':  # Windows
                venv_pytest = os.path.join(self.venv_path, "Scripts", "pytest.exe")
            else:  # Unix-like systems
                venv_pytest = os.path.join(self.venv_path, "bin", "pytest")
            
            result = subprocess.run([venv_pytest, "--version"], capture_output=True, text=True, check=True)
            if result.returncode == 0:
                log(f"Pytest version in the virtual environment: {result.stdout.strip()}", "INFO")
            else:
                log(f"Error checking Pytest version: {result.stderr.strip()}", "ERROR")
        except subprocess.CalledProcessError as e:
            log(f"Error checking Pytest version: {e.stderr.strip()}", "ERROR")
        except Exception as e:
            log(f"Error executing Pytest in the virtual environment: {e}", "ERROR")
            raise

    def install_dependency(self, package_name):
        """
        Installs a Python package into the specified virtual environment.

        Args:
            package_name (str): The name of the Python package to install.
        """
        pip_executable = os.path.join(self.venv_path, "bin", "pip")
        packages_to_install = [
            ("pip", "setuptools"),
            ("pytest",),
            ("pytest-json-report",),
            ("pytest-cov",),
            ("pydantic",)
        ]
        for package_tuple in packages_to_install:
            try:
                subprocess.run([pip_executable, "install", "--quiet", "-U"] + list(package_tuple), check=True)
            except subprocess.CalledProcessError as e:
                log(f"Error installing {' '.join(package_tuple)}: {e}", "ERROR")
                raise

        try:
            subprocess.run([pip_executable, "install", "--quiet", "-U"] + package_name.split(), check=True)
            log(f"Package '{package_name}' has been installed in the virtual environment.", "INFO")
        except subprocess.CalledProcessError as e:
            log(f"Error installing {package_name}: {e}", "ERROR")
            raise

    def sanitize_path(path):
        """
        Sanitize a path to be used in the PATH environment variable.
        """
        # Remove any characters that aren't alphanumeric, '-', '_', '/', or '.'
        return ''.join(c for c in path if c.isalnum() or c in '-_/.')

    def activate_venv(self):
        """
        Activates the specified virtual environment by modifying the current process's environment.
        
        Returns:
            bool: True if activation was successful, False otherwise.
        """
        if sys.platform.startswith("win"):
            activate_script = os.path.join(self.venv_path, "Scripts", "activate.bat")
            try:
                # Use a list of arguments instead of a string to avoid shell=True
                activate_command = [os.environ.get("COMSPEC", "cmd.exe"), "/c", activate_script, "&&", "set"]
                process = subprocess.Popen(
                    activate_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                stdout, stderr = process.communicate()
                
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, activate_command, stderr)

                for line in stdout.splitlines():
                    if '=' not in line:
                        continue
                    key, value = line.split('=', 1)
                    if key.upper() == 'PATH':
                        # Sanitize PATH components
                        value = os.pathsep.join(sanitize_path(p) for p in value.split(os.pathsep))
                    os.environ[key] = value
            except subprocess.CalledProcessError as e:
                log(f"Failed to activate virtual environment '{self.venv_path}': {e}", "ERROR")
                return False
        else:
            # On Unix-like systems, we'll manually modify the environment variables
            try:
                # Modify PATH
                bin_path = sanitize_path(os.path.join(self.venv_path, "bin"))
                current_path = os.environ.get("PATH", "")
                new_path = os.pathsep.join([bin_path] + [sanitize_path(p) for p in current_path.split(os.pathsep) if p])
                os.environ["PATH"] = new_path
                
                # Set VIRTUAL_ENV
                os.environ["VIRTUAL_ENV"] = sanitize_path(self.venv_path)
                
                # Unset PYTHONHOME if it's set
                os.environ.pop("PYTHONHOME", None)
                
                # Prepend the virtual env's Python to sys.path
                python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
                site_packages = sanitize_path(os.path.join(self.venv_path, "lib", python_version, "site-packages"))
                sys.path.insert(0, site_packages)
            except Exception as e:
                log(f"Failed to activate virtual environment '{self.venv_path}': {e}", "ERROR")
                return False

        # Verify activation by checking VIRTUAL_ENV environment variable
        if os.environ.get("VIRTUAL_ENV") == sanitize_path(self.venv_path):
            log(f"Virtual environment '{self.venv_path}' has been activated.", "INFO")
            return True
        else:
            log(f"Failed to activate virtual environment '{self.venv_path}'.", "ERROR")
            return False

class PythonExecutor:
    def __init__(self, venv_path):
        self.venv_path = os.path.abspath(venv_path)
        # self.venv_path = venv_path
        self.activate_script = os.path.join(self.venv_path, "bin", "activate")
        if sys.platform.startswith("win"):
            self.activate_script = os.path.join(self.venv_path, "Scripts", "activate.bat")

    def activate_venv(self):
        """
        Activates the specified virtual environment.

        Args:
            venv_path (str): The path to the virtual environment directory.
        """
        if sys.platform.startswith("win"):
            activate_script = os.path.join(self.venv_path, "Scripts", "activate.bat")
            return_code = subprocess.run(shlex.split(f"call {self.activate_script}"))
        else:
            return_code = subprocess.run(shlex.split(f"source {self.activate_script}"))
        if return_code == 0:
            log(f"Virtual environment '{self.venv_path}' has been activated.", "INFO")
            return True
        else:
            log(f"Failed to activate virtual environment '{self.venv_path}'.", "ERROR")
            return False
        
    def get_python_path(self):
        try:
            if os.name == 'nt':  # Windows
                venv_python = os.path.join(self.venv_path, "Scripts", "python.exe")
            else:  # Unix-like systems
                venv_python = os.path.join(self.venv_path, "bin", "python")
            result = subprocess.run([venv_python, "-c", "import sys; print(sys.executable)"], capture_output=True, text=True)
            if result.returncode == 0:
                log(f"Python path in the virtual environment: {result.stdout.strip()}", "DEBUG")
                return result.stdout.strip()
            else:
                log(f"Error checking Python path: {result.stderr.strip()}", "ERROR")
        except Exception as e:
            log(f"Error executing Python in the virtual environment: {e}", "ERROR")

    def run_pytest(self, file_path):
        """
        Runs pytest in the specified virtual environment.

        Args:
            file_path (str): The path to the file containing the pytest tests.
        """
        result = None
        if os.name == 'nt':  # Windows
            venv_python = os.path.join(self.venv_path, "Scripts", "python.exe")
            venv_pytest = os.path.join(self.venv_path, "Scripts", "pytest.exe")
        else:  # Unix-like systems
            venv_python = os.path.join(self.venv_path, "bin", "python")
            venv_pytest = os.path.join(self.venv_path, "bin", "pytest")
        # venv_python path above also worked. just in case
        venv_python = self.get_python_path()
        try:
            # Create a temporary file for the JSON report
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as tmp:
                json_report_path = tmp.name

            command = [
                venv_python, 
                "-m", 
                "pytest", 
                # "-v", # Verbose output
                # "-s", # Print captured output
                "--json-report",
                f"--json-report-file={json_report_path}",
                file_path
            ]
            # print(f'command {command}')

            # Set up the environment
            env = os.environ.copy()
            env["VIRTUAL_ENV"] = self.venv_path 
            result = subprocess.run(command,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    universal_newlines=True,
                                    env=env,
                                    check=False # allow handling of output when test fails 
                                    )
            
            # Read and parse the JSON report file (could not directly load it as josn string)
            with open(json_report_path, 'r') as json_file:
                json_output = json.load(json_file)
                log(f'filename={file_path}, json_output: {json_output}', "DEBUG")
            # Clean up the temporary file
            os.unlink(json_report_path)

            return self.parse_json_report(json_output)
        except Exception as e:
            return {"error": str(e)}
 
    def run_coverage(self, file_path: str = 'tests/'):
        """
        Runs pytest in the specified virtual environment.

        Args:
            file_path (str): The path to the file containing the pytest tests.
        """
        result = None
        if os.name == 'nt':  # Windows
            venv_python = os.path.join(self.venv_path, "Scripts", "python.exe")
            venv_pytest = os.path.join(self.venv_path, "Scripts", "pytest.exe")
        else:  # Unix-like systems
            venv_python = os.path.join(self.venv_path, "bin", "python")
            venv_pytest = os.path.join(self.venv_path, "bin", "pytest")
        # venv_python path above also worked. just in case
        venv_python = self.get_python_path()
        try:
            command = [
                venv_python, 
                "-m", 
                "pytest", 
                "--cov=.",  # Replace with the path to your package
                "--cov-report=term-missing",
                "--cov-report=json",   # Display a report with missing lines
                # "--cov-config=<(echo -e '[run]\nomit = tests/conftest.py, main.py, *__init__.py')",
                # Additional options for coverage report
                # "--cov-report=html",         # Generate HTML report
                # "--cov-report=xml",          # Generate XML report
                file_path
            ]
            # print(f'command {command}')

            # Set up the environment
            env = os.environ.copy()
            env["VIRTUAL_ENV"] = self.venv_path 
            result = subprocess.run(command,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    universal_newlines=True,
                                    env=env,
                                    check=False # allow handling of output when test fails 
                                    )
            # print(f'result {result}')
            # Read and parse the JSON report file (could not directly load it as josn string)
            json_report_path = 'coverage.json'
            with open(json_report_path, 'r') as json_file:
                json_output = json.load(json_file)
                log(f'filename={json_report_path}, json_output: {json_output}', "DEBUG")
            return self.parse_coverage_report(json_output)
        except Exception as e:
            return {"error": str(e)}

    def parse_coverage_report(self, coverage_data):
        """
        Parses the coverage report data and returns a dictionary with relevant information.

        Args:
            coverage_data (dict): The coverage report data in JSON format.

        Returns:
            dict: A dictionary containing the coverage summary and details for each file.
        """
        coverage_report = {}

        # Extract metadata
        metadata = coverage_data.get("meta", {})
        coverage_report["metadata"] = {
            "format": metadata.get("format"),
            "version": metadata.get("version"),
            "timestamp": metadata.get("timestamp"),
            "branch_coverage": metadata.get("branch_coverage"),
            "show_contexts": metadata.get("show_contexts"),
        }

        # Extract coverage summary
        coverage_report["summary"] = {
            "total_statements": 0,
            "covered_statements": 0,
            "missing_statements": 0,
            "percent_covered": 0.0,
        }

        # Extract file-level coverage details
        coverage_report["files"] = {}

        for file_path, file_data in coverage_data.get("files", {}).items():
            file_summary = file_data.get("summary", {})
            coverage_report["files"][file_path] = {
                "num_statements": file_summary.get("num_statements", 0),
                "covered_lines": file_summary.get("covered_lines", 0),
                "missing_lines": file_summary.get("missing_lines", []),
                "percent_covered": file_summary.get("percent_covered", 0.0),
            }

            coverage_report["summary"]["total_statements"] += file_summary.get("num_statements", 0)
            coverage_report["summary"]["covered_statements"] += file_summary.get("covered_lines", 0)
            missing_lines = file_summary.get("missing_lines", [])
            if isinstance(missing_lines, list):
                coverage_report["summary"]["missing_statements"] += len(missing_lines)
            else:
                coverage_report["summary"]["missing_statements"] += missing_lines

        if coverage_report["summary"]["total_statements"] > 0:
            coverage_report["summary"]["percent_covered"] = (
                coverage_report["summary"]["covered_statements"] / coverage_report["summary"]["total_statements"]
            ) * 100

        return coverage_report
    
    def parse_json_report(self, json_output):
        """
        Parses the JSON output from a test report and extracts the summary and failed test information.

        Args:
            json_output (dict): The JSON output from the test report.

        Returns:
            dict: A dictionary containing the summary and failed test information.
        """        
        result_dict = {
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "errors": 0,
                "duration": 0,
            },
            "failed_tests": [],
            "collection_errors": []
        }
        
        if 'summary' in json_output:
            summary = json_output['summary']
            result_dict["summary"] = {
                "total": summary.get('total', 0),
                "passed": summary.get('passed', 0),
                "failed": summary.get('failed', 0),
                "skipped": summary.get('skipped', 0),
                "errors": summary.get('errors', 0),
                "duration": summary.get('duration', 0)
            }
            
        if 'collectors' in json_output:
            for collector in json_output['collectors']:
                if 'errors' in collector:
                    for error in collector['errors']:
                        result_dict["collection_errors"].append({
                            "file": collector['nodeid'],
                            "error_message": error.get('longrepr', 'Unknown error during collection')
                        })
        if 'tests' in json_output:
            for test in json_output['tests']:
                if test['outcome'] == 'failed':
                    result_dict["failed_tests"].append({
                        "file": test['nodeid'].split("::")[0],
                        "test": test['nodeid'].split("::")[-1],
                        "status": "FAILED",
                        "error_message": test['call'].get('longrepr', 'No detailed error message available')
                    })
        
        return result_dict
    
    def find_error_message(self, message: str):
        """
        Extracts the error message from the given string.

        Args:
            message (str): The string to search for an error message.

        Returns:
            str: The extracted error message, or None if no error message is found.
        """
        lines = message.split('\n')
        error_patterns = ['Error:', 'Exception:']
        for pattern in error_patterns:
            for line in message.split('\n'):
                if pattern in line:
                    return line.strip()
        return None
    
    def exec_python_file(self, filename):
        '''
        This approach does not replicate the actual python file execution
        '''
        test_passed = False
        error_message = ''
        stderr = ''
        stdout = ''
        try:
            python_executable = os.path.join(self.venv_path, "bin", "python")
            command = [python_executable, f'{filename}']
            result = subprocess.run(command,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   universal_newlines=True)
            stdout = result.stdout
            stderr = result.stderr
        except subprocess.CalledProcessError as e:
            log(f"command failed with exit code: {e.returncode}", "INFO")
            self.handle_subprocess_error(e)
            stdout = e.stdout
            stderr = e.stderr
        else:
            self.handle_result(result)
            if result.returncode == 0:
                test_passed = True
        error_message = self.find_error_message(stderr) or self.find_error_message(stdout)
        return {
            "stdout": stdout,
            "stderr": stderr,
            "test_passed": test_passed,
            "error_message": error_message
        }

    def get_current_dir(self):
        return os.getcwd()
    
    def exec_python_string(self, py_string):
        test_passed = False
        error_message = ''
        stderr = ''
        stdout = ''
        temp_file = None
        try:
            # Create a temporary file and write the provided code to it
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
                temp_file.write(py_string)
                temp_file.flush()

            python_executable = os.path.join(self.venv_path, "bin", "python")
            command = [python_executable, temp_file.name]

            project_path = self.get_current_dir()
            env = os.environ.copy()
            env['PYTHONPATH'] = f"{project_path}:{env.get('PYTHONPATH', '')}"
            result = subprocess.run(command,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   universal_newlines=True,
                                   cwd=project_path,
                                   env=env)

            stdout = result.stdout
            stderr = result.stderr

            if result.returncode == 0:
                test_passed = True
            else:
                error_message = self.find_error_message(stderr) or self.find_error_message(stdout)
        except subprocess.CalledProcessError as e:
            log(f"command failed with exit code: {e.returncode}", "INFO")
            self.handle_subprocess_error(e)
            stdout = e.stdout
            stderr = e.stderr
            error_message = self.find_error_message(stderr) or self.find_error_message(stdout)
        finally:
            # Clean up the temporary file
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
        return {
            "stdout": stdout,
            "stderr": stderr,
            "test_passed": test_passed,
            "error_message": error_message
        }

    def handle_result(self, result):
        if result.returncode != 0:
            self.log(f"command failed with exit code: {result.returncode}", "INFO")
            if result.stderr:
                print(result.stderr.strip())
            elif result.stdout:
                print(result.stdout.strip())
            else:
                print("No output from the command.")
        else:
            self.log("command succeeded", "INFO")
            if result.stderr:
                print(result.stderr.strip())
            elif result.stdout:
                print(result.stdout.strip())
            else:
                print("No output from the command.")

    def handle_subprocess_error(self, e):
        if e.stderr:
            print(e.stderr.strip())
        elif e.stdout:
            print(e.stdout.strip())
        else:
            print("No output from the command.")

    def handle_result(self, result):
        if result.returncode != 0:
            print(f"command failed with exit code: {result.returncode}")
            if result.stderr:
                print(result.stderr.strip())
            elif result.stdout:
                print(result.stdout.strip())
            else:
                print("No output from the command.")
        else:
            print("command succeeded")
            if result.stderr:
                print(result.stderr.strip())
            elif result.stdout:
                print(result.stdout.strip())
            else:
                print("No output from the command.")