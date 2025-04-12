import setuptools
import os
import io

_PROJECT_NAME = "pixel-paladin-rl"
_PROJECT_VERSION = "0.1.0"
_PROJECT_AUTHOR = "Omdeep Borkar"
_PROJECT_AUTHOR_EMAIL = "omdeeborkar@gmail.com"
_PROJECT_URL = "https://github.com/Omdeepb69/Pixel Paladin RL"
_PROJECT_DESCRIPTION = (
    "Train an AI agent using Reinforcement Learning to master a "
    "custom-built Pygame environment, learning complex strategies "
    "beyond simple scripted behavior. It's dangerous to go alone; train this!"
)
_PYTHON_REQUIRES = ">=3.8"
_README_FILE = "README.md"
_SRC_DIR = "src" # Assumes source code is in a 'src' directory

def read_readme(file_name=_README_FILE):
    """Reads the README file for the long description."""
    try:
        # Use io.open for encoding specification and handle path correctly
        readme_path = os.path.join(os.path.dirname(__file__), file_name)
        if not os.path.exists(readme_path):
            # Fallback if README is not present at the expected location
            print(f"Warning: README file not found at {readme_path}")
            return _PROJECT_DESCRIPTION # Fallback to short description

        with io.open(readme_path, encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Warning: Could not read README file: {e}")
        return _PROJECT_DESCRIPTION # Fallback to short description

_CORE_REQUIREMENTS = [
    "pygame>=2.1.0",
    "numpy>=1.21.0",
    "matplotlib>=3.5.0",
]

_EXTRA_REQUIREMENTS = {
    "tf": ["tensorflow>=2.8.0"],
    "pytorch": ["torch>=1.10.0"],
    "gym": ["gymnasium>=0.26.0"],
    "dev": [
        "pytest>=7.0.0",
        "flake8>=4.0.0",
        "black>=22.0.0",
        "mypy>=0.930",
        "isort>=5.10.0",
        "pip-tools>=6.0.0",
    ],
}
_EXTRA_REQUIREMENTS["all_agents"] = (
    _EXTRA_REQUIREMENTS["tf"] + _EXTRA_REQUIREMENTS["pytorch"]
)
_EXTRA_REQUIREMENTS["all"] = list(
    set(
        _EXTRA_REQUIREMENTS["tf"]
        + _EXTRA_REQUIREMENTS["pytorch"]
        + _EXTRA_REQUIREMENTS["gym"]
        + _EXTRA_REQUIREMENTS["dev"]
    )
)


setuptools.setup(
    name=_PROJECT_NAME,
    version=_PROJECT_VERSION,
    author=_PROJECT_AUTHOR,
    author_email=_PROJECT_AUTHOR_EMAIL,
    description=_PROJECT_DESCRIPTION,
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url=_PROJECT_URL,
    packages=setuptools.find_packages(where=_SRC_DIR),
    package_dir={"": _SRC_DIR},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Games/Entertainment",
    ],
    python_requires=_PYTHON_REQUIRES,
    install_requires=_CORE_REQUIREMENTS,
    extras_require=_EXTRA_REQUIREMENTS,
    project_urls={
        "Bug Tracker": f"{_PROJECT_URL}/issues",
        "Source Code": _PROJECT_URL,
        "Documentation": _PROJECT_URL, # Placeholder, update if docs exist
    },
    include_package_data=True,
    # entry_points={ # Uncomment and define if you have console scripts
    #     'console_scripts': [
    #         'pixel-paladin-train=pixel_paladin_rl.main:train',
    #         'pixel-paladin-play=pixel_paladin_rl.main:play',
    #     ],
    # },
)