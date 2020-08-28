import re
import json
import pandas as pd
from os.path import dirname, join

NAMES = [
    ("universidad del norte", "colombia"),
    ("universidad nacional de colombia", "colombia"),
    ("universidad de antioquia", "colombia"),
    ("universidad industrial de santander", "colombia"),
    ("universidad del valle", "colombia"),
    ("universidad del cauca", "colombia"),
    ("instituto tecnologico metropolitano", "colombia"),
]


def extract_country_name(x):
    #
    if pd.isna(x) or x is None:
        return pd.NA
    ##
    ## List of standardized country names
    ##
    module_path = dirname(__file__)
    with open(join(module_path, "../data/worldmap.data"), "r") as f:
        countries = json.load(f)
    country_names = list(countries.keys())

    ##
    ## Adds missing countries to list of
    ## standardized countries
    ##
    for name in ["Singapore", "Malta", "United States"]:
        country_names.append(name)

    ##
    ## Country names to lower case
    ##
    country_names = {country.lower(): country for country in country_names}

    ##
    ## Replace administrative regions by country names
    ## for the current string
    ##
    x = x.lower()
    x = x.strip()
    for a, b in [
        ("bosnia and herzegovina", "bosnia and herz."),
        ("brasil", "brazil"),
        ("czech republic", "czechia"),
        ("espana", "spain"),
        ("hong kong", "china"),
        ("macao", "china"),
        ("macau", "china"),
        ("peoples r china", "china"),
        ("rusia", "russia"),
        ("russian federation", "russia"),
        ("united states of america", "united states"),
        ("usa", "united states"),
    ]:
        x = re.sub(a, b, x)

    ##
    ## Name search in the affiliation (x)
    ##
    for z in reversed(x.split(",")):

        z = z.strip()

        ##
        ## Exact match in list of stadardized country names
        ##
        if z.lower() in country_names.keys():
            return country_names[z.lower()]

        ##
        ## Discard problems of multiple blank spaces
        ##
        z = " ".join([w.strip() for w in z.lower().split(" ")])
        if z in country_names.keys():
            return country_names[z]

    ##
    ## Repair country name from institution name
    ##
    for institution, country in NAMES:
        if institution in x:
            return country

    ##
    ## Country not found
    ##
    return pd.NA
