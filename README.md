# passive-radar
Capstone Project (ECE496) @ UofT

## Setting up GNU Radio and SoapySDR with Conda

This guide provides step-by-step instructions for setting up GNU Radio and SoapySDR in Python using Conda environments.

### Prerequisites

- Python 3.8 or higher
- Conda or Miniconda installed on your system
  - Download from: https://docs.conda.io/en/latest/miniconda.html
- Basic knowledge of command-line operations

### System Requirements

- **Operating System**: Linux (Ubuntu/Debian recommended), macOS, or Windows with WSL2
- **RAM**: Minimum 4GB (8GB+ recommended)
- **Disk Space**: At least 5GB free space for Conda environment and packages

### Step 1: Create a Conda Environment

Create a new Conda environment for the passive radar project:

```bash
conda create -n passive-radar python=3.10
```

Activate the environment:

```bash
conda activate passive-radar
```

### Step 2: Install GNU Radio

Install GNU Radio from the conda-forge channel:

```bash
conda install -c conda-forge gnuradio
```

Verify the installation:

```bash
gnuradio-config-info --version
```

### Step 3: Install SoapySDR

Install SoapySDR and its Python bindings:

```bash
conda install -c conda-forge soapysdr
conda install -c conda-forge soapysdr-module-all
```

### Step 4: Install SDR Hardware Support (Optional)

Depending on your SDR hardware, install the appropriate drivers:

#### For RTL-SDR:
```bash
conda install -c conda-forge soapysdr-module-rtlsdr
```

#### For HackRF:
```bash
conda install -c conda-forge soapysdr-module-hackrf
```

#### For USRP (Ettus Research):
```bash
conda install -c conda-forge soapysdr-module-uhd
conda install -c conda-forge uhd
```

#### For Airspy:
```bash
conda install -c conda-forge soapysdr-module-airspy
```

### Step 5: Install Additional Python Packages

Install commonly needed packages for signal processing:

```bash
conda install -c conda-forge numpy scipy matplotlib
```

### Verification

Verify your installation by running the following commands:

1. Check GNU Radio installation:
```bash
python -c "import gnuradio; print(gnuradio.__version__)"
```

2. Check SoapySDR installation:
```bash
SoapySDRUtil --info
```

3. List available SDR devices:
```bash
SoapySDRUtil --find
```

4. Launch GNU Radio Companion (GUI):
```bash
gnuradio-companion
```

### Troubleshooting

#### Issue: Conda packages conflict
**Solution**: Create a fresh environment and install packages in the order specified above.

#### Issue: SDR device not detected
**Solution**: 
- Ensure device drivers are properly installed
- Check USB connections and permissions (Linux users may need to add udev rules)
- Verify device compatibility with SoapySDR

#### Issue: GNU Radio Companion won't start
**Solution**: 
- Ensure you have X11 forwarding enabled (for remote sessions)
- Check that all dependencies are installed: `conda install -c conda-forge gtk3 pygobject`

#### Issue: Import errors in Python
**Solution**: Make sure the Conda environment is activated before running Python scripts.

### Alternative: Using environment.yml

For easier setup, you can create an `environment.yml` file with all dependencies:

```yaml
name: passive-radar
channels:
  - conda-forge
dependencies:
  - python=3.10
  - gnuradio
  - soapysdr
  - soapysdr-module-all
  - numpy
  - scipy
  - matplotlib
```

Then create the environment with:

```bash
conda env create -f environment.yml
conda activate passive-radar
```

### Updating Packages

To update all packages in your environment:

```bash
conda update -c conda-forge --all
```

### Deactivating the Environment

When finished working:

```bash
conda deactivate
```

### Removing the Environment

To completely remove the environment:

```bash
conda env remove -n passive-radar
```

### Next Steps

- Test your SDR hardware with SoapySDR utilities
- Explore GNU Radio examples: Run `gnuradio-companion` and open example flowgraphs
- Start developing your passive radar signal processing chains

### Additional Resources

- GNU Radio Documentation: https://www.gnuradio.org/doc/
- SoapySDR Wiki: https://github.com/pothosware/SoapySDR/wiki
- Conda Documentation: https://docs.conda.io/
