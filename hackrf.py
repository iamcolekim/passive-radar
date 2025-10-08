# HackRF One SDR using SoapySDR and cross-ambiguity function for signal processing

import SoapySDR
from SoapySDR import *  # SOAPY_SDR_ constants
import numpy as np


def assert_device_count(results, required_count=2):
    if len(results) < required_count:
        raise AssertionError(f"Error: {required_count} HackRF devices are required, found {len(results)}.")

def configure_device(device, sample_rate, center_freq, gain):
    try:
        device.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, sample_rate)
        device.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, center_freq)
        device.setGain(SoapySDR.SOAPY_SDR_RX, 0, gain)
    except Exception as e:
        raise RuntimeError(f"Failed to configure SDR device: {e}")

def setup_stream(device):
    try:
        stream = device.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
        stream.activate()
    except Exception as e:
        raise RuntimeError(f"Failed to setup/activate SDR stream: {e}")
    return stream

def read_samples(stream, num_samples, timeout_us):
    buff = np.zeros(num_samples, dtype=np.complex64)
    try:
        stream.read(buff, num_samples, timeoutUs=timeout_us)
    except Exception as e:
        raise RuntimeError(f"Failed to read samples from SDR stream: {e}")
    return buff

def cleanup(device, stream):
    try:
        stream.deactivate()
        device.closeStream(stream)
    except Exception as e:
        print(f"Warning: Error during cleanup: {e}")



def main():
    # Find all available SDR devices
    device = SoapySDR.Device
    results = device.enumerate()
    assert_device_count(results, required_count=2)

    # suggested parameters, feel free to modify
    sample_rate = 10e6
    center_freq = 100e6
    gain = 20
    num_samples = 1024
    timeout_us = 1000000

    # Setup devices
    device_ref = setup_device(results[0])
    device_surveil = setup_device(results[1])

    # Configure devices
    configure_device(device_ref, sample_rate, center_freq, gain)
    configure_device(device_surveil, sample_rate, center_freq, gain)

    # Setup streams
    stream_ref = setup_stream(device_ref)
    stream_surveil = setup_stream(device_surveil)

    # Read samples
    buff_ref = read_samples(stream_ref, num_samples, timeout_us)
    buff_surveil = read_samples(stream_surveil, num_samples, timeout_us)

    # Cleanup
    cleanup(device_ref, stream_ref)
    cleanup(device_surveil, stream_surveil)

    # Now you have your two signal arrays: buff_ref and buff_surveil
    # You can proceed with the cross-ambiguity function as planned.

if __name__ == "__main__":
    main()