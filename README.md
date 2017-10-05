# Resample High Frequency Data

## Introduction
These codes resample high frequency financial data using a
large variety of published approaches.

## Supported Data

[Trades and Quotes](http://www.nyxdata.com/Data-Products/Daily-TAQ)
[Forex](https://www.tickdata.com/product/historical-forex-data/)


## Approaches

Currently, these techniques are implemented:
* Calendar Time Sampling (CTS)
* Transaction Time Sampling (TrTS)
* Tick Time Sampling (TTS)
* Duration Adjustment Sampling (DA)
* Time Transformation Sampling (TT)
* Business Time Sampling (BTS)
* A smoothed Business Time Sampling variant (sBTS)
* Weighted Standard Deviation Sampling (WSD)

The names in brackets refer to the variables in the process_data.py file.
For references to these approaches see the comments within the code.


## Usage

See [my repo on realized quantities](https://github.com/BayerSe/RealizedQuantities)
