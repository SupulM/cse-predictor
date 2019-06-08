## Stock Price trend prediction for Colombo Stock Exchange

Uses data from [cse.lk](https://www.cse.lk) and Long short-term memory (LSTM) neural networks implemented using Python with the [Keras](https://keras.io) library.

### Pre-requisites

Python and pip should be installed in your system.

### Install and run instructions

Clone the project and `cd` into it, run the following command to download required libraries. This will create the `venv` folder in the project location.

Installing `virtualenv` for the Virtual Python Environment
```bash
pip install virtualenv
```
```bash
virtualenv venv
```

Activating the Virtual Environment
* MacOS / Linux
```bash
source venv/bin/activate
```
* Windows
```bash
venv\Scripts\activate
```
Installing the required libraries for the project
```bash
pip install -r requirements.txt
```
Run the training using data from the `datasets` folder. The company ID is the 4-letter code.
```bash
python CSE_train.py COMB
```

Run the testing using saved models from the `saved` folder. The company ID is the 4-letter code.
```bash
python CSE_test.py COMB
```

