# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 16:47:15 2023

@author: carlo
"""

import pandas as pd
import numpy as np
import re, os, sys
import time
from datetime import datetime
from string import punctuation
from decimal import Decimal
from operator import itemgetter
from fuzzywuzzy import fuzz, process

import bq_functions
import gspread

import matplotlib.colors as mcolors

import json
import doctest # run via doctest.testmod()

## REQUIREMENTS
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
output_path = os.path.abspath(os.path.dirname(__file__))
os.chdir(output_path) # current working directory

platform_gsheet = {'GULONG' : {'url' : 'https://docs.google.com/spreadsheets/d/1mHsdtKhdQkm0wals7wn_A5jOicf1WOBuonXEpF8bW0U/edit#gid=0'
                            },
                   'CARMAX' : {'url' : 'https://docs.google.com/spreadsheets/d/1SUBvR4UbpGzkMsxl_TXY7du45LKMDGP3_IHNd3tCzSY/edit#gid=1686244668',
                            }
                   }

def read_gsheet(url, title):
    
    with open('credentials.txt') as f:
        data = f.read()

    credentials = json.loads(data)    
    
    gsheet_key = re.search('(?<=\/d\/).*(?=\/edit)', url)[0]
    gc = gspread.service_account_from_dict(credentials)
    wb = gc.open_by_key(gsheet_key)
    
    try:
        sh = wb.worksheet(title)
        df = pd.DataFrame.from_records(sh.get_all_records())
    except:
        df = None
    
    return df

def lev_dist(seq1, seq2):
    '''
    Calculates levenshtein distance between texts
    
    >>> lev_dist('Manila', 'MANILA')
    5
    >>> lev_dist('ARIVO', 'APOLLO')
    4
    '''
    
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return int((matrix[size_x - 1, size_y - 1]))

def remove_emoji(text):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, ' ', text).strip()


def punctuation_removal(punc):
    '''
    Removes needed symbols from punctuation list (which will be removed from model strings)
    '''
    remove_list = ['&']
    
    for r in remove_list:
        try:
            ndx = punc.index(r)
            punc = punc[:ndx] + punc[ndx+1:]
        except:
            continue
    return punc

def get_best_match(query, match_list):
    '''
    Get best match of query to a list of matches
    
    query : str
        string looking for match
    match_list: list of results from process.extractBests()
    
    '''
    
    # if no matches, return NaN
    if len(match_list) == 0:
        return np.NaN
    # if only one match, return match
    elif len(match_list) == 1:
        return match_list[0][0]
    
    # if multiple matches
    else:
        # get match strings and corresponding scores
        matches, scores = list(zip(*match_list))
        # check if any match has a high enough score
        if any((best_index := scores.index(s)) for s in scores if s >= 95):
            return matches[best_index]
        # if no match score is high enough, get match with lowest lev dist
        else:
            min_lev_dist = 100
            best_match = np.NaN
            for m in matches:
                lev_d = lev_dist(query, m)
                if lev_d < min_lev_dist:
                    best_match = m
                    min_lev_dist = lev_d
                else:
                    continue
                
            return best_match

def import_makes(platform):
    '''
    Import list of makes
    '''
    # output_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
    #                                                                '..'))
    
    if any((match := key) == platform.strip().upper() for key in list(platform_gsheet.keys())):
           df = read_gsheet(platform_gsheet[match]['url'], 
                            'brands')
           df.columns = ['_'.join(c.lower().strip().split(' ')) for c in df.columns]
    else:
        raise Exception('Wrong platform keyword.')
    
    return df

def clean_make(x, makes):
    '''
    Cleans Car makes input
    
    Parameters
    ----------
    x : string
        makes string input
    makes: dataframe
        dataframe of reference makes with string as first column
        second columns contains aliases
    
    >>> clean_make('PORSHE', carmax_makes)
    'PORSCHE'
    >>> clean_make('PUEGEOT', carmax_makes)
    'PEUGEOT'
    >>> clean_make('VW', carmax_makes)
    'VOLKSWAGEN'
    >>> clean_make('MORRIS GARAGES', carmax_makes)
    'MG'
    
    Notes:
        Need to accommodate makes_list input as list datatype
    
    '''
    def flatten(l):
        return [item for sublist in l for item in sublist]
    
    # check makes input datatype
    makes_ = makes.fillna('').copy()
    makes_list = makes_.iloc[:, 0].str.upper().tolist()
    makes_.iloc[:, 1] = makes_.iloc[:, 1].str.upper()
    alias_list = flatten([i.split(', ') for i in makes_.iloc[:, 1].value_counts().index])
    
    search_thresh = 80
    exact_thresh = 95 # for MORRIS GARAGES
    # check if input is not NaN
    if pd.notna(x):
        # baseline correct
        x = str(x).strip().upper()
        # check main brands list
        matches = process.extractBests(x, makes_list, score_cutoff = search_thresh)
        # check if any exact match with brands
        try:
            ## if exact match with brand names, skip alias check
            if sum(z for z in list(list(zip(*matches))[1]) if z >= exact_thresh):
                pass
            else:
                ## proceed to alias check
               raise Exception('Exact match not found in makes list')
        except:       
            alias_matches = process.extractBests(x, alias_list, score_cutoff = search_thresh)
            # get original brand of matched alias
            matches += [(makes_[makes_.iloc[:, 1].str.contains(n[0])].iloc[:,0].values[0], n[1]) for n in alias_matches if n[1] >= search_thresh]

        finally:
            try:
                best_match = max(matches, key = itemgetter(1))[0].strip().upper()
                
            except:
                best_match = np.NaN
            
            finally:
                return best_match
        
    else:
        return np.NaN

def import_models(platform, makes = None):
    '''
    Import list of models
    
    '''

    if makes is not None:
        pass
    else:
        makes = import_makes(platform)
        
    models_sh_list = []
    for make in makes.name:
        time.sleep(1.75) # reduce failed imports due to timeout
        try:
            models_df = read_gsheet(platform_gsheet[platform]['url'], 
                                              make)
            models_df.loc[:, 'make'] = make
            models_sh_list.append(models_df)
            print(f'Added {make}')
        except:
            print (f'Failed to add {make}')
            continue

    df = pd.concat(models_sh_list, axis=0, ignore_index = True)
    df.columns = ['_'.join(c.strip().lower().split(' ')) for c in df.columns]
    return df


def clean_model(model, makes, models):
    '''
    Cleans carmax model string
    
    Parameters
    ----------
    model : string
    makes : dataframe
    models : dataframe
    
    Returns
    -------
    cleaned model string
    
    >>> clean_models('2015 MITSUBISHI ASX GLS AT', carmax_makes, carmax_models)
    'ASX'
    >>> clean_models('2022 Peugeot 2008 1.2 THP Allure AT', carmax_makes, carmax_models)
    '2008'
    >>> clean_models('2017 BMW X3 AT Diesel', carmax_makes, carmax_models)
    'X3'
    >>> clean_models('2015 Toyota Vios 1.3 E MT Gasoline', carmax_makes, carmax_models)
    'VIOS'
    '''
    
    def flatten(l):
        return [item for sublist in l for item in sublist]
    
    # check makes input datatype
    makes_ = makes.fillna('').copy()
    makes_list = makes_.iloc[:, 0].str.upper().str.strip().tolist()
    makes_.iloc[:, 1] = makes_.iloc[:, 1].str.upper().str.strip()
    makes_alias_list = flatten([i.split(', ') for i in makes_.iloc[:, 1].value_counts().index])
    
    search_thresh = 80
    exact_thresh = 95 # for MORRIS GARAGES
    
    if pd.notna(model) and (model is not None):
        model = re.sub('[(\t)(\n)]', ' ', model.upper().strip())
        
        brand_matches = process.extractBests(model, makes_list, score_cutoff = search_thresh)
        # check if any exact match with brands
        try:
            ## if exact match with brand names, skip alias check
            if sum(z for z in list(list(zip(*brand_matches))[1]) if z >= exact_thresh):
                pass
            else:
                ## proceed to alias check
               raise Exception('Exact match not found in makes list')
        except:       
            alias_matches = process.extractBests(model, makes_alias_list, score_cutoff = search_thresh)
            # get original brand of matched alias
            brand_matches += [(makes_[makes_.iloc[:, 1].str.contains(n[0])].iloc[:,0].values[0], n[1]) for n in alias_matches if n[1] >= search_thresh]        
        
        #best_brand_match = max(brand_matches, key = itemgetter(1))[0].strip().upper()
        best_brand_match = get_best_match(model, brand_matches)
        
        if best_brand_match:
            models_ = models[models.make.str.upper() == best_brand_match].fillna('')
        else:
            models_ = models.fillna('').copy()
        
        ## put columns in variables for easy access
        models_list = models_.iloc[:, 0].str.upper().tolist()
        models_.iloc[:, 1] = models_.iloc[:,1].str.upper().str.strip()
        models_alias_list = flatten([i.split(', ') for i in models_.iloc[:, 1].value_counts().index])
        
        if len(models_list):
            pass
        else:
            return model
            
        ## fuzzy search for model from filtered model list
        model_matches = process.extractBests(model, models_list, score_cutoff = search_thresh)
        
        try:
            ## if exact match with brand names, skip alias check
            if sum(z for z in list(list(zip(*model_matches))[1]) if z >= exact_thresh):
                pass
            else:
                ## proceed to alias check
               raise Exception('Exact match not found in makes list')
        except:       
            models_alias_matches = process.extractBests(model, models_alias_list, score_cutoff = search_thresh)
            ## get original brand of matched alias
            model_matches += [(models_[models_.iloc[:, 1].str.contains(n[0])].iloc[:,0].values[0], n[1]) for n in models_alias_matches if n[1] >= search_thresh]        
        
        ## find best model match from obtained matches
        finally:
            try:
                #best_match = max(model_matches, key = itemgetter(1))[0].strip().upper()
                best_match = get_best_match(model, model_matches)
                ## if brand was found, remove brand from result
                if pd.notna(best_match):
                    if best_brand_match:
                        best_match = re.sub(best_brand_match, '', best_match).strip()
                    else:
                        pass
                else:
                    raise Exception('Match not found.')
                
            except:
                best_match = np.NaN
            
            finally:
                return best_match
        
    else:
        return np.NaN

## START OF CARMAX FUNCTIONS ================================================##

def import_body_type(platform):
    
    '''
    Import list of body types

    '''
    
    if any((match := key) == platform.strip().upper() for key in list(platform_gsheet.keys())):
           df = read_gsheet(platform_gsheet[match]['url'], 
                            'body_type')
           df.columns = ['_'.join(c.lower().strip().split(' ')) for c in df.columns]
    else:
        raise Exception('Wrong platform keyword.')
    
    return df

def clean_body_type(x, body_types):
    '''
    Clean vehicle/body types using fuzzywuzzy partial ratio (if needed)
    
    See import_body_type
    '''
    
    def flatten(l):
        return [item for sublist in l for item in sublist]
    
    body = body_types.fillna('').copy()
    body_list = body.iloc[:, 0].str.upper().tolist()
    body.iloc[:, 1] = body.iloc[:, 1].str.upper()
    alias_list = flatten([i.split(', ') for i in body.iloc[:, 1].value_counts().index])
    
    if pd.notna(x):
        # baseline correction
        x = str(x).strip().upper()
        
        # check main brands list
        best_matches = []
        best_matches += [n for n in process.extractBests(x, body_list) if n[1] >= 85]
        
        # check if any exact match with brands
        try:
            if sum(z for z in list(list(zip(*best_matches))[1]) if z >= 95):
                pass
            else:
               raise Exception('Exact match not found in makes list')
               
        except:       
            alias_matches = process.extractBests(x, alias_list)
            # get original brand of matched alias
            best_matches += [(body[body.iloc[:, 1].str.contains(n[0])].iloc[:,0].values[0], n[1]) for n in alias_matches if n[1] >= 85]

        finally:
            try:
                best_match = max(best_matches, key = itemgetter(1))[0].strip().upper()
                
            except:
                best_match = np.NaN
            
            finally:
                return best_match        
        
    else:
        return np.NaN

def import_colors():
    '''
    Imports reference list of colors
    
    Required import:
        import matplotlib.colors as mcolors

    '''
    color1 = pd.Series(list(mcolors.CSS4_COLORS.keys())).str.upper().tolist()
    color2 = [c.split('xkcd:')[-1].upper().strip() for c in mcolors.XKCD_COLORS.keys()]
    all_colors = pd.Series(color1 + color2).drop_duplicates().tolist()
    return all_colors

def clean_color(c, colors):
    
    '''
    Cleans color value string of carmax entries
    
    See import_colors
    '''
    # returns '' if NaN
    if pd.notna(c):
        # baseline correction
        c = c.upper().strip()
        matches = process.extractBests(c, colors)
        best_match = get_best_match(c, matches)
        return best_match
    else:
        return np.NaN

def import_locations():
    '''
    Imports philippine locations in terms of city/municipality, province, region
    '''
    
    df = read_gsheet(platform_gsheet['CARMAX']['url'], 
                     'ph_locations')
    df.columns = ['_'.join(c.lower().strip().split(' ')) for c in df.columns]
    
    return df

def clean_location(loc, ph_loc, prov = None):
    
    city_dict = {'QC' : 'Quezon City'}
    
    
    if pd.isna(loc):
        return np.NaN, np.NaN, np.NaN
    else:
        loc = loc.title().strip()
        if ('City' in loc.split(', ')[0]) and (loc.split(', ')[0] in ph_loc[ph_loc.city.str.contains('City')]['city'].unique()):
            pass
        elif ('City' in loc.split(', ')[0]):
            loc = ', '.join([loc.split(', ')[0].split('City')[0].strip()] + loc.split(', ')[1:])
        # Check cities first
        
        if any((match := city_dict[l]) for l in city_dict.keys() if process.extractOne(l, loc.split(', '))[1] >= 85):
            city_match = match
        else:
            city_match_list = []
            for l in loc.split(', '):
                bests = process.extractBests(l, ph_loc.city)
                for b in bests:
                    if b[1] >= 75:
                        city_match_list.append(b)
            
            #city_match_list = [f[0] for f in fuzzy_city_match if f[1] >= 85]
            if len(city_match_list) > 0:
                city_match = get_best_match(loc, city_match_list)
            else:
                city_match = np.NaN
        
        if pd.notna(city_match):
            if prov is not None:
                prov_match = process.extractOne(prov, ph_loc.province)[0]
            else:
                prov_match = ph_loc[ph_loc.city == city_match]['province'].iloc[0]
                
            region_match = ph_loc[(ph_loc.city == city_match) & (ph_loc.province == prov_match)]['region'].iloc[0]
            
        else:
            if prov is not None:
                prov_match = process.extractOne(prov, ph_loc.province)[0]
            else:
                fuzzy_prov_match = process.extractBests(loc, ph_loc.province)
                prov_match_list = [f for f in fuzzy_prov_match if f[1] >= 80]
                prov_match = get_best_match(loc, prov_match_list)
            
            if pd.notna(prov_match):
                region_match = ph_loc[ph_loc.province == prov_match]['region'].iloc[0]
            
            else:
                fuzzy_region_match = process.extractBests(loc, ph_loc.region)
                region_match_list = [f for f in fuzzy_region_match if f[1] >= 85]
                region_match = get_best_match(loc, region_match_list)
        
        return city_match, prov_match, region_match

def clean_year(x):
    '''
    Finds the year in a string value
    '''
    if pd.isna(x):
        return x
    else:
        x = str(x).strip().upper()
        x = re.search('(19|20)[0-9]{2}', x)
        if x:
            return x[0]
        else:
            return np.NaN

def clean_price(x):
    '''
    Cleans price string values from carmax entries
    
    >>> clean_price('P123,456')
    123456.0
    >>> clean_price('₱123,456.25')
    123456.25
    >>> clean_price('₱1.125 MILLION')
    1125000.0
    >>> clean_price('P1.125M')
    1125000.0
    >>> clean_price(123456)
    123456.0
    '''
    if pd.isna(x):
        return np.NaN
    else:
        # baseline correct
        x = str(x).upper().strip()
        # normal result
        try:
            if 'MILLION' in x or 'M' in x:
                match = re.search('[1-9](.)?[0-9]+((?<!MILLION)|(?<!M))', x)
                return float(match[0])*1E6
            else:
                match = re.search('((?<=P)|(?<=₱))?(\s)?[1-9]+(,)?[0-9]+(,)?[0-9]+(.)?[0-9]+',x)
                return float(''.join(match[0].strip().split(',')))
        # unexpected result
        except:
            # get all digits
            try:
                return float(''.join(re.findall('[0-9]', x)))
            # return cleaned string
            except:
                return np.NaN

def clean_mileage(x, description = None):
    '''
    Cleans mileage values
    
    >>> clean_mileage('45,000 KM')
    45000.0
    >>> clean_mileage('45,000 MILEAGE')
    45000.0
    >>> clean_mileage('45K KM')
    45000.0
    >>> clean_mileage('45K MILEAGE')
    45000.0
    
    '''
    if pd.isna(x) or (x is None):
        if (description is not None) and pd.notna(description):
            description = description.strip().upper()
            r = re.search('[0-9]+(,)?[0-9]+K?(\s)?((?=MILEAGE)|(?=KM))?', description)
            if r is not None:
                if 'K' in r[0]:
                    return float(''.join(re.findall('[0-9]', r[0])))*1E3
                else:
                    return float(''.join(r[0].strip().split(',')))
            else:
                return np.NaN
        else:
            return np.NaN
    else:
        # baseline correction
        x = str(x).upper().strip()
        # normal result
        try:
            r = re.search('[0-9]+(,)?[0-9]+K?(\s)?((?=MILEAGE)|(?=KM))?', x)
            if r is not None:
                if 'K' in r[0]:
                    return float(''.join(re.findall('[0-9]', r[0])))*1E3
                else:
                    return float(''.join(r[0].strip().split(',')))
            else:
                if (description is not None) and pd.notna(description):
                    description = description.strip().upper()
                    r = re.search('[0-9]+(,)?[0-9]+K?(\s)?((?=MILEAGE)|(?=KM))?', description)
                    if r is not None:
                        if 'K' in r[0]:
                            return float(''.join(re.findall('[0-9]', r[0])))*1E3
                        else:
                            return float(''.join(r[0].strip().split(',')))
                    else:
                        return np.NaN
                else:
                    raise Exception
        # unexpected result
        except:
            # get all digits
            try:
                return float(''.join(re.findall('[0-9]', x)))
            # return NaN
            except:
                return np.NaN

def clean_fuel_type(x, description = None):
    '''
    Cleans fuel_type data
    
    Parameters
    ----------
    x : string
        fuel_type string input or NaN
    
    DOCTESTS:
    >>> clean_fuel_type('GAS')
    'GASOLINE'
    >>> clean_fuel_type('DEISEL')
    'DIESEL'
    >>> clean_fuel_type('GASSOLIN')
    'GASOLINE'
    >>> clean_fuel_type('DISEL')
    'DIESEL'
    '''
    f_dict = {'GAS' : 'GASOLINE', 
                 'DIESEL' : 'DIESEL',
                 'PETROL': 'PETROL',
                 'FLEX': 'FLEX/E85',
                 'E85' : 'FLEX/E85',
                 'ELECTRIC' : 'ELECTRIC'}
    
    def regex_fuel(s):
        s = str(s).strip().upper()
        match_val = process.extractOne(s, f_dict.values())
        match_key = process.extractOne(s, f_dict.keys())
        if match_val[1] > 50:
            return match_val[0]
        elif match_key[1] > 50:
            return f_dict[match_key[0]]
        else:
            raise Exception
    
    if pd.isna(x) or (x is None):
        try:
            if (description is not None):
                return regex_fuel(description)
            else:
                raise Exception
        except:
            return np.NaN
    else:
        x = str(x).upper().strip()
        try:
            matches = process.extractBests(x, list(f_dict.values()))
            best_match = get_best_match(x, matches)
            if best_match is not None:
                return best_match
            else:
                raise Exception('Best match not found.')
        
        except:
            return x
        
def clean_transmission(x, variant = None, description = None):
    '''
    Cleans/Finds transmission string values from carmax data
    
    DOCTESTS:
    >>> clean_transmission('AT')
    'AUTOMATIC'
    >>> clean_transmission('MANUEL')
    'MANUAL'
    >>> clean_transmission('MT')
    'MANUAL'
    >>> clean_transmission('ELEC')
    'ELECTRIC'
    >>> clean_transmission('VARIABLE')
    'CVT'
    '''
    t_dict = {'AT' : 'AUTOMATIC',
              'MT' : 'MANUAL',
              'CVT' : 'CVT',
              'VARIABLE' : 'CVT',
              'DUAL-CLUTCH' : 'DCT',
              'AUTO' : 'AUTOMATIC',
              'ELECTRIC' : 'ELECTRIC'}
    
    def regex_trans(s):
        s = str(s).strip().upper()
        match_val = process.extractOne(s, t_dict.values())
        match_key = process.extractOne(s, t_dict.keys())
        if match_val[1] > 50:
            return match_val[0]
        elif match_key[1] > 50:
            return t_dict[match_key[0]]
        
        else:
            t = re.search('(A|M|CV)\/?T', s)
            if t is not None:
                return t_dict[''.join(t[0].split('/'))]
            else:
                raise Exception
    
    # value is NaN
    if pd.isna(x) or (x is None):
        # check if car variant has transmission
        try:
            return regex_trans(variant)
        except:
            try:
                return regex_trans(description)
            except:
                return np.NaN
    # value is not NaN
    else:
        # baseline correct
        x = str(x).upper().strip()
        try:
            # if exact match
            if x in t_dict.values():
                return x
            
            # subset match in values
            elif any((match := t) for t in t_dict.values() if fuzz.partial_ratio(x, t) >= 85):
                return match
            
            # subset match in keys
            elif any((match := v) for v in t_dict.keys() if fuzz.partial_ratio(x, v) >= 85):
                return t_dict[match]
            
            else:
                match = process.extractOne(x, t_dict.values())
                if match[1] < 60:
                    raise Exception
                else:
                    return match[0]
              
        except:
            try:
                return regex_trans(variant)
            except:
                try:
                    return regex_trans(description)
                except:
                    return x

def clean_engine(x, description = None):
    '''
    Cleans engine string value from carmax entries
    '''
    if pd.isna(x):
        if description is not None:
            x = description.upper().strip()
            if (match := re.search('(?<![0-9])[0-9]\.[0-9]((?<=L)|(?<=-LITER))?', x)):
                return f'{float(match[0].strip())}L'
            else:
                return np.NaN
        else:
            return np.NaN
        
    else:
        # baseline correction
        x = str(x).upper().strip()
        if (match := re.search('(?<![0-9])[0-9]\.[0-9]((?<=L)|(?<=-LITER))?', x)):
            return f'{float(match[0].strip())}L'
        else:
            if description is not None:
                x = description.upper().strip()
                if (match := re.search('(?<![0-9])[0-9]\.[0-9]((?<=L)|(?<=-LITER))?', x)):
                    return f'{float(match[0].strip())}L'
                else:
                    return np.NaN
            else:
                return np.NaN
            

def clean_engine_disp(x):
    if pd.notna(x):
        return '{:.1f}'.format(round(float(x)/1000, 2)) + 'L'
    else:
        return np.NaN

## END OF CARMAX FUNCTIONS ==================================================##

## START OF GULONG FUNCTIONS ================================================##

def combine_specs(w, ar, d, mode = 'SKU'):
    '''
    
    Parameters
    - w: string
        section_width
    - ar: string
        aspect_ratio
    - d: string
        diameter
    - mode: string; optional
        SKU or MATCH
    
    Returns
    - combined specs with format for SKU or matching
    
    >>> combine_specs('175', 'R', 'R15', mode = 'SKU')
    '175/R15'
    >>> combine_specs('175', '65', 'R15', mode = 'SKU')
    '175/65/R15'
    >>> combine_specs('33', '12.5', 'R15', mode = 'SKU')
    '33X12.5/R15'
    >>> combine_specs('LT175', '65', 'R15C', mode = 'SKU')
    'LT175/65/R15C'
    >>> combine_specs('LT175', '65', 'R15C', mode = 'MATCH')
    '175/65/15'
    >>> combine_specs('175', '65', '15', mode = 'SKU')
    '175/65/R15'
    
    '''
    
    if mode == 'SKU':
        d = d if 'R' in d else 'R' + d 
        if ar != 'R':
            if '.' in ar:
                return w + 'X' + ar + '/' + d
            else:
                return '/'.join([w, ar, d])
        else:
            return w + '/' + d
            
    elif mode == 'MATCH':
        w = ''.join(re.findall('[0-9]|\.', str(w)))
        ar = ''.join(re.findall('[0-9]|\.|R', str(ar)))
        d = ''.join(re.findall('[0-9]|\.', str(d)))
        return '/'.join([w, ar, d])

    else:
        combine_specs(str(w), str(ar), str(d), mode = 'SKU')

def fix_names(sku_name, comp=None):
    '''
    Fix product names to match competitor names
    
    Parameters
    ----------
    sku_name: str
        input SKU name string
    comp: list (optional)
        optional list of model names to compare with
    
    Returns
    -------
    name: str
        fixed names as UPPERCASE
    '''
    
    # replacement should be all caps
    change_name_dict = {'TRANSIT.*ARZ.?6-X' : 'TRANSITO ARZ6-X',
                        'TRANSIT.*ARZ.?6-A' : 'TRANSITO ARZ6-A',
                        'TRANSIT.*ARZ.?6-M' : 'TRANSITO ARZ6-M',
                        'OPA25': 'OPEN COUNTRY A25',
                        'OPA28': 'OPEN COUNTRY A28',
                        'OPA32': 'OPEN COUNTRY A32',
                        'OPA33': 'OPEN COUNTRY A33',
                        'OPAT\+': 'OPEN COUNTRY AT PLUS', 
                        'OPAT2': 'OPEN COUNTRY AT 2',
                        'OPMT2': 'OPEN COUNTRY MT 2',
                        'OPAT OPMT': 'OPEN COUNTRY AT',
                        'OPAT': 'OPEN COUNTRY AT',
                        'OPMT': 'OPEN COUNTRY MT',
                        'OPRT': 'OPEN COUNTRY RT',
                        'OPUT': 'OPEN COUNTRY UT',
                        'DC -80': 'DC-80',
                        'DC -80+': 'DC-80+',
                        'KM3': 'MUD-TERRAIN T/A KM3',
                        'KO2': 'ALL-TERRAIN T/A KO2',
                        'TRAIL-TERRAIN T/A' : 'TRAIL-TERRAIN',
                        '265/70/R16 GEOLANDAR 112S': 'GEOLANDAR A/T G015',
                        '265/65/R17 GEOLANDAR 112S' : 'GEOLANDAR A/T G015',
                        '265/65/R17 GEOLANDAR 112H' : 'GEOLANDAR G902',
                        'GEOLANDAR A/T 102S': 'GEOLANDAR A/T-S G012',
                        'GEOLANDAR A/T': 'GEOLANDAR A/T G015',
                        'ASSURACE MAXGUARD SUV': 'ASSURANCE MAXGUARD SUV',
                        'EFFICIENTGRIP SUV': 'EFFICIENTGRIP SUV',
                        'EFFICIENGRIP PERFORMANCE SUV':'EFFICIENTGRIP PERFORMANCE SUV',
                        'WRANGLE DURATRAC': 'WRANGLER DURATRAC',
                        'WRANGLE AT ADVENTURE': 'WRANGLER AT ADVENTURE',
                        'WRANGLER AT ADVENTURE': 'WRANGLER AT ADVENTURE',
                        'WRANGLER AT SILENT TRAC': 'WRANGLER AT SILENTTRAC',
                        'ENASAVE  EC300+': 'ENSAVE EC300 PLUS',
                        'SAHARA AT2' : 'SAHARA AT 2',
                        'SAHARA MT2' : 'SAHARA MT 2',
                        'WRANGLER AT SILENT TRAC': 'WRANGLER AT SILENTTRAC',
                        'POTENZA RE003 ADREANALIN': 'POTENZA RE003 ADRENALIN',
                        'POTENZA RE004': 'POTENZA RE004',
                        'SPORT MAXX 050' : 'SPORT MAXX 050',
                        'DUELER H/T 470': 'DUELER H/T 470',
                        'DUELER H/T 687': 'DUELER H/T 687 RBT',
                        'DUELER A/T 697': 'DUELER A/T 697',
                        'DUELER A/T 693': 'DUELER A/T 693 RBT',
                        'DUELER H/T 840' : 'DUELER H/T 840 RBT',
                        'EVOLUTION MT': 'EVOLUTION M/T',
                        'BLUEARTH AE61' : 'BLUEARTH XT AE61',
                        'BLUEARTH ES32' : 'BLUEARTH ES ES32',
                        'BLUEARTH AE51': 'BLUEARTH GT AE51',
                        'COOPER STT PRO': 'STT PRO',
                        'COOPER AT3 LT' : 'AT3 LT',
                        'COOPER AT3 XLT' : 'AT3 XLT',
                        'A/T3' : 'AT3',
                        'ENERGY XM+' : 'ENERGY XM2+',
                        'XM2+' : 'ENERGY XM2+',
                        'AT3 XLT': 'AT3 XLT',
                        'ADVANTAGE T/A DRIVE' : 'ADVANTAGE T/A DRIVE',
                        'ADVANTAGE T/A SUV' : 'ADVANTAGE T/A SUV'
                        }
    
    if pd.isna(sku_name) or (sku_name is None):
        return np.NaN
    
    else:
        # uppercase and remove double spaces
        raw_name = re.sub('  ', ' ', sku_name).upper().strip()
        # specific cases
        for key in change_name_dict.keys():
            if re.search(key, raw_name):
                return change_name_dict[key]
            else:
                continue
        
        # if match list provided
        
        if comp is not None:
            # check if any name from list matches anything in sku name
            match_list = [n for n in comp if re.search(n, raw_name)]
            # exact match from list
            if len(match_list) == 1:
                return match_list[0]
            # multiple matches (i.e. contains name but with extensions)
            elif len(match_list) > 1:
                long_match = ''
                for m in match_list:
                    if len(m[0]) > len(long_match):
                        long_match = m[0]
                return long_match
            # no match
            else:
                return raw_name
        else:
            return raw_name

def remove_trailing_zero(num):
    '''
    Removes unnecessary zeros from decimals

    Parameters
    ----------
    num : Decimal(number)
        number applied with Decimal function (see import decimal from Decimal)

    Returns
    -------
    number: Decimal
        Fixed number in Decimal form

    '''
    return num.to_integral() if num == num.to_integral() else num.normalize()

def clean_width(w, model = None):
    '''
    Clean width values
    
    Parameters
    ----------
    d: string
        width values in string format
        
    Returns:
    --------
    d: string
        cleaned diameter values
    
    DOCTESTS:
    >>> clean_width('7')
    '7'
    >>> clean_width('175')
    '175'
    >>> clean_width('6.50')
    '6.5'
    >>> clean_width('27X')
    '27'
    >>> clean_width('LT35X')
    'LT35'
    >>> clean_width('8.25')
    '8.25'
    >>> clean_width('P265.5')
    'P265.5'
    >>> clean_width(np.NaN)
    nan
    
    '''
    if pd.notna(w):
        w = str(w).strip().upper()
        # detects if input has expected format
        prefix_num = re.search('[A-Z]*[0-9]+.?[0-9]*', w)
        if prefix_num is not None:
            num_str = ''.join(re.findall('[0-9]|\.', prefix_num[0]))
            num = str(remove_trailing_zero(Decimal(num_str)))
            prefix = w.split(num_str)[0]
            return prefix + num
        else:
            return np.NaN
    else:
        if model is None:
            return np.NaN
        else:
            try:
                width = model.split('/')[0].split(' ')[-1].strip().upper()
                return clean_width(width)   
            except:
                return np.NaN

def clean_diameter(d):
    '''
    Fix diameter values
    
    Parameters
    ----------
    d: string
        diameter values in string format
        
    Returns:
    --------
    d: string
        fixed diameter values
    
    DOCTESTS:
    >>> clean_diameter('R17LT')
    'R17LT'
    >>> clean_diameter('R22.50')
    'R22.5'
    >>> clean_diameter('15')
    'R15'
    >>> clean_diameter(np.NaN)
    nan
    
    '''
    if pd.notna(d):
        d = str(d).strip().upper()
        num_suffix = re.search('[0-9]+.?[0-9]*[A-Z]*', d)
        if num_suffix is not None:
            num_str = ''.join(re.findall('([0-9]|\.)', num_suffix[0]))
            num = str(remove_trailing_zero(Decimal(num_str)))
            suffix = num_suffix[0].split(num_str)[-1]
            return f'R{num}{suffix}'
    else:
        return np.NaN

def clean_aspect_ratio(ar, model = None):
    
    '''
    Clean raw aspect ratio input
    
    Parameters
    ----------
    ar: float or string
        input raw aspect ratio data
    model: string; optional
        input model string value of product
        
    Returns
    -------
    ar: string
        fixed aspect ratio data in string format for combine_specs
    
    DOCTESTS:
    >>> clean_aspect_ratio('/')
    'R'
    >>> clean_aspect_ratio('.5')
    '9.5'
    >>> clean_aspect_ratio('14.50')
    '14.5'
    >>> clean_aspect_ratio(np.NaN)
    'R'
    
    '''
    error_ar = {'.5' : '9.5',
                '0.': '10.5',
                '2.': '12.5',
                '3.': '13.5',
                '5.': '15.5',
                '70.5': '10.5'}
    
    if pd.notna(ar):
        # aspect ratio is faulty
        if str(ar) in ['0', 'R1', '/', 'R']:
            return 'R'
        # incorrect parsing osf decimal aspect ratios
        elif str(ar) in error_ar.keys():
            return error_ar[str(ar)]
        # numeric/integer aspect ratio
        elif str(ar).isnumeric():
            return str(ar)
        # alphanumeric
        elif str(ar).isalnum():
            return ''.join(re.findall('[0-9]', str(ar)))
        
        # decimal aspect ratio with trailing 0
        elif '.' in str(ar):
            return str(remove_trailing_zero(Decimal(str(ar))))
        
        else:
            return np.NaN
        
    else:
        return 'R'

def clean_speed_rating(sp):
    '''
    Clean speed rating of gulong products
    
    DOCTESTS:
    >>> clean_speed_rating('W XL')
    'W'
    >>> clean_speed_rating('0')
    'B'
    >>> clean_speed_rating('118Q')
    'Q'
    >>> clean_speed_rating('T/H')
    'T'
    >>> clean_speed_rating('-')
    nan
    
    '''

    # not NaN
    if pd.notna(sp):
        # baseline correct
        sp = sp.strip().upper()
        # detect if numerals are present 
        num = re.search('[0-9]{2,3}', sp)
        
        if num is None:
            pass
        else:
            # remove if found
            sp = sp.split(num[0])[-1].strip()
            
        if 'XL' in sp:
            return sp.split('XL')[0].strip()
        elif '/' in sp:
            return sp.split('/')[0].strip()
        elif sp == '0':
            return 'B'
        elif sp == '-':
            return np.NaN
        else:
            return sp
    else:
        return np.NaN

def combine_sku(make, w, ar, d, model, load, speed):
    '''
    DOCTESTS:
            
    >>> combine_sku('ARIVO', '195', 'R', 'R15', 'TRANSITO ARZ 6-X', '106/104', 'Q')
    'ARIVO 195/R15 TRANSITO ARZ 6-X 106/104Q'
    
    '''
    specs = combine_specs(w, ar, d, mode = 'SKU')
    
    if (load in ['nan', np.NaN, None, '-', '']) or (speed in ['nan', np.NaN, None, '', '-']):
        return ' '.join([make, specs, model])
    else:
        return ' '.join([make, specs, model, load + speed])

def promo_GP(price, cost, sale_tag, promo_tag):
    '''
    price : float
        price_gulong price
    cost : float
        supplier cost price
    sale_tag : binary
        buy 4 tires 3% off per tire
    promo_tag : binary
        buy 3 tires get 1 free
        
    DOCTESTS:
    >>> promo_GP(4620, 3053.65, 1, 0)
    5711
    >>> promo_GP(5040, 3218.4, 0, 1)
    2246.4
        
    '''
    # sale tag : buy 4 tires 3% off per tire
    # promo tag: buy 3 tires get 1 free
    if sale_tag:
        gp = (price * 0.97 - cost) * 4
    else:
        gp = (price * 3 - cost * 4)
    return round(gp, 2)

def calc_overall_diameter(specs):
    '''
    # width cut-offs ; 5 - 12.5 | 27 - 40 | 145 - 335
    # aspect ratio cut-off : 25
    # diameter issues: none
    
    >>> calc_overall_diameter('225/35/18')
    24.2
    >>> calc_overall_diameter('7.5/R/16')
    31.7
    >>> calc_overall_diameter('35/12.5/17')
    35.0
    
    1710: 12.5/80/16 -> 38.5/12.5/16
    827: 12.5/80/16 -> 36/12.5/16
    810: 12.5/80/15 -> 38/12.5/16
    804: 10.5/80/15 -> 31/10.5/15
    '''
    w, ar, d = specs.split('/')
    w = float(w)
    ar = 0.82 if ar == 'R' else float(ar)/100.0
    d = float(d)
    
    if w <= 10:
        return round((w + 0.35)*2+d, 2)
    
    elif 10 < w < 25:
        return np.NaN
    
    elif 27 <= w <= 60:
        return round(w, 1)
    
    elif w > 120:
        return round((w*ar*2)/25.4 + d, 2)
    
    else:
        return np.NaN

def get_car_compatible():
    
    # http://app.redash.licagroup.ph/queries/183
    # GULONG - Car Compatible Tire Sizes
    def import_data():
        print ('Importing database data')
        url = 'http://app.redash.licagroup.ph/api/queries/183/results.csv?api_key=NWVzsgA5xGzhpW4xhslaJ5Nlx9o1ghM7P5a9PtHb'
        comp_data = pd.read_csv(url, parse_dates = ['created_at', 'updated_at'])
    
        print ('Importing makes and models list')
        main('CARMAX', 'CLEAN')
        
        print ('Cleaning data')
        comp_data.loc[:, 'car_make'] = comp_data.apply(lambda x: clean_make(x['car_make'], carmax_makes), axis=1)
        comp_data.loc[:, 'car_model'] = comp_data.apply(lambda x: ' '.join(x['car_model'].split('-')).upper(), axis=1)
        
        comp_data.loc[:, 'width'] =  comp_data.apply(lambda x: clean_width(x['section_width']), axis=1)
        comp_data.loc[:, 'aspect_ratio'] = comp_data.apply(lambda x: clean_aspect_ratio(x['aspect_ratio']), axis=1)
        comp_data.loc[:, 'diameter'] = comp_data.apply(lambda x: clean_diameter(x['rim_size']), axis=1)
        comp_data.loc[:, 'correct_specs'] = comp_data.apply(lambda x: combine_specs(x['width'], x['aspect_ratio'], x['diameter'], mode = 'MATCH'), axis=1)
        #comp_data.to_csv('car_comp_tire_size.csv')
        return comp_data
    
    start_time = time.time()
    print ('Start car comparison tire size data import')

    comp_data = import_data()
    
    #comp_data.loc[:,'year'] = comp_data.year.astype(str)
    print ('Imported tire size car comparison data')
    print("--- %s seconds ---" % (time.time() - start_time))
    
    return comp_data

## END OF GULONG FUNCTIONS ==================================================##

def clean_df(df):

#     try:
#         temp = df[df.date.notna()]
#         temp.loc[:, 'date'] = temp.date.dt.date
#         df = pd.concat([df[df.date.isnull()], temp], axis = 0).sort_values('date', 
#                                                                            ascending = False)\
#                                                                 .reset_index(drop = True)
#     except:
#         pass
    
#     df.loc[:, 'model'] = df.apply(lambda x: clean_models(x['model'], makes_list, models_list) 
#                                   if pd.notna(x['model']) else clean_model(x['url'], makes_list, models_list), axis=1)
    
#     df.loc[:, 'make'] = df.apply(lambda x: clean_makes(x['make'], makes_list)
#                                  if pd.notna(x['make']) else clean_makes(x['url'], makes_list), axis=1)
    
#     df = df[df.model.notna()]
    
#     # year
#     df = df[df.year.notna()]
#     df.loc[:, 'year'] = df.apply(lambda x: int(clean_year(x['year'])), axis=1)
#     df = df[df.year.between(2000, datetime.today().year - 1)]
    
#     # transmission
#     df.loc[:, 'transmission'] = df.apply(lambda x: clean_transmission(x['transmission'], variant = ' '.join(x['url'].split('-')).upper()), axis=1)
#     df = df[df.transmission.isin(['AUTOMATIC', 'MANUAL'])]
    
#     # mileage
#     df.loc[:, 'mileage'] = df.apply(lambda x: clean_mileage(x['mileage'], description = x['description']), axis=1)
#     ## mileage outlier removal per year
#     mileage_q = df.groupby('year')['mileage'].describe().loc[:, ['25%', '50%', '75%']]
#     mileage_q.loc[:,'IQR'] = (mileage_q.loc[:,'75%'] - mileage_q.loc[:, '25%'])
#     mileage_q.loc[:, 'upper'] = mileage_q.loc[:, '75%'] + 1.5*mileage_q.loc[:,'IQR']
#     df.loc[:, 'mileage_check'] = df.apply(lambda x: 1 if x['mileage'] <= mileage_q.loc[x['year'], 'upper'] else 0, axis=1)
#     df = df[(df.mileage_check == 1) & (df.mileage >= 3000)]
    
#     # fuel_type
#     df.loc[:, 'fuel_type'] = df.apply(lambda x: clean_fuel_type(x['fuel_type'], description = x['description']), axis=1)
#     df = df[df.fuel_type.isin(['GASOLINE', 'DIESEL'])]
    
#     # price
#     df.loc[:, 'price'] = df.apply(lambda x: clean_price(x['price']), axis=1)
#     df = df[df.price <= 2000000]
    
#     price_q = df.groupby('year')['price'].describe().loc[:, ['25%', '50%', '75%']]
#     price_q.loc[:, 'IQR'] = (price_q.loc[:, '75%'] - price_q.loc[:, '25%'])
#     price_q.loc[:, 'upper'] = price_q.loc[:, '75%'] + 1.5*price_q.loc[:, 'IQR']
#     df.loc[:, 'price_check'] = df.apply(lambda x: 1 if x['price'] <= price_q.loc[x['year'], 'upper'] else 0, axis=1)
#     df = df[(df.price_check == 1) & (df.price >= 50000)]
    
#     # num_photos
#     df.loc[:, 'num_photos'] = df.num_photos.apply(int)
    
#     # body_type
#     df.loc[:, 'body_type'] = df.body_type.apply(lambda x: clean_body_type(x, body_types))
#     df = df[df.body_type.isin(['SUV', 'SEDAN', 'HATCHBACK', 'PICKUP TRUCK', 'VAN'])]
    
#     # location
#     df['city'], df['province'], df['region'] = zip(*df.location.apply(lambda x: clean_location(x, ph_locations)))
    
#     # remove duplicates
#     df = df.drop_duplicates(subset  = ['make', 'model', 'year', 'mileage', 
#                                        'transmission'],
#                             keep = 'first')
#     # remove na
#     df = df.dropna(subset  = ['make', 'model', 'year', 'mileage', 
#                                        'transmission']).reset_index(drop = True)
    
#     df = df.drop(['mileage_check', 'price_check', 'description', 'location'], axis=1)
#     df = df.rename(columns = {'date': 'date_posted'})
    
    return df

def prep_competitor_data(platform):
    
    '''
    Imports competitor data from BQ and performs data cleaning
    Uploads cleaned competitor data to BQ
    '''
    
    ## get GCP BQ credentials
    with open('secrets.json') as s:
        acct = json.load(s)
    
    ## load BQ client
    client, credentials = bq_functions.authenticate_bq(acct)
    project_id = credentials.project_id
    
    ## TODO : Query carmax competitor raw data from BQ
    if platform is not None:
        ## Carmax
        if platform.upper() == 'CARMAX':
            
            competitor_list = ['autodeal', 'tsikot', 'philkotse', 'usedcarsphil']
            df = pd.concat([bq_functions.query_bq('.'.join([project_id, 'carmax', c]), client) 
                            for c in competitor_list])
            
            ## TODO : Clean competitor data
            df_clean = clean_df(df)
            
            ## TODO : Write cleaned data to BQ/GCS
            bq_functions.bq_write(df_clean, credentials, 'carmax', 'competitor_data', client)
            
        else:
            ## Gulong
            pass
        
        return df_clean
    
    
    

def main(platform = None, purpose = None):
    '''
    Run initialization functions for variables depending on platform and purpose
    to reduce runtime on unnecessary functions
    '''
    if (platform is not None) and (purpose is not None):
        if (platform.upper() == 'GULONG') and (purpose == 'CLEAN'):
            global gulong_makes
            gulong_makes = import_makes('GULONG')
            
        elif (platform.upper() == 'CARMAX') and (purpose == 'CLEAN'):
            global carmax_makes, carmax_models, body_types
            carmax_makes = import_makes('CARMAX')
            carmax_models = import_models('CARMAX', 
                                          carmax_makes).replace('', np.NaN)
            body_types = import_body_type('CARMAX')
        else:
            pass
    else:
        pass
    
    if all(True for v in ['punctuation', 'colors', 'ph_locations'] if v in globals()):
        pass
    
    else:
        global punctuation, colors, ph_locations
        punctuation = punctuation_removal(punctuation)
        colors = import_colors()
        ph_locations = import_locations()
    

if __name__ == '__main__':
    main()
    ## functions testing
    #doctest.testmod()