# Project Outline

A set of scripts to use machine learning to predict the non-linear (and non-sine) long term frequency drift cause by temperature changes inside the cavity in the lab. This will then be used to update a temperature control box ahead of time so that the cavity is the correct temperature at the correct time.


## Usage

Simply clone the repo and change the paths in `Scripts/predictCavityDrift.py`
As an example of code usage, I have included a set of pre-sorted cavity data in the pickled file: `cavdat_20200701-20201005`. Additionally, I have included examples of the .lvm and geography build temperature data files in the `Example LVM data` folder.
  
  
### `sortLVM.py`
This script contains functions used to take the data stored in the lvm files that the Single Atom Lab LabVIEW Interface (SALLI) generates and store it into a pandas dataframe. To save on ram I would recomend setting `sample_period` (which defines the minimum number of seconds present between samples) to 600 (i.e. 1 sample point every 10 minutes).
An example of the data which has been pre-loaded and pickled into `cavdat_20200701-20201005` is shown below:  
![Test Image](Figures/uncleaned_data.png)




## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.



## License
[MIT](https://choosealicense.com/licenses/mit/)



## Version info:
This is a list of the packages and their versions I had installed while writing these scripts:  
  
INSTALLED VERSIONS  
------------------  
python           : 3.8.3.final.0  
python-bits      : 64  
OS               : Windows  
OS-release       : 10  
machine          : AMD64  
processor        : Intel64 Family 6 Model 60 Stepping 3, GenuineIntel  

pandas           : 1.0.5  
numpy            : 1.18.5  
pytz             : 2020.1  
dateutil         : 2.8.1  
pip              : 20.1.1  
setuptools       : 49.2.0.post20200714  
Cython           : 0.29.21  
pytest           : 5.4.3  
sphinx           : 3.1.2  
xlsxwriter       : 1.2.9  
lxml.etree       : 4.5.2  
html5lib         : 1.1  
jinja2           : 2.11.2  
IPython          : 7.16.1  
bs4              : 4.9.1  
bottleneck       : 1.3.2  
lxml.etree       : 4.5.2  
matplotlib       : 3.2.2  
numexpr          : 2.7.1  
openpyxl         : 3.0.4  
pytest           : 5.4.3  
scipy            : 1.5.0  
sqlalchemy       : 1.3.18  
tables           : 3.6.1  
xlrd             : 1.2.0  
xlwt             : 1.3.0  
xlsxwriter       : 1.2.9  
numba            : 0.50.1  
sklearn          : 0.23.1  