# -*- coding: utf-8 -*-
"""
Created on Fri May 19 12:27:14 2023

@author: carlo
"""

import pandas as pd
import numpy as np
import time

import config_lica as config
import bq_functions

import streamlit as st
from st_aggrid import GridOptionsBuilder, AgGrid
import extra_streamlit_components as stx

       
@st.cache_data
def get_gulong_data():
    '''
    Get gulong.ph data from backend
    
    Returns
    -------
    df : dataframe
        Gulong.ph product info dataframe
    '''
    #df = pd.read_csv('http://app.redash.licagroup.ph/api/queries/130/results.csv?api_key=JFYeyFN7WwoJbUqf8eyS0388PFE7AiG1JWa6y9Zp')
    # http://app.redash.licagroup.ph/queries/131
    url1 =  "http://app.redash.licagroup.ph/api/queries/131/results.csv?api_key=FqpOO9ePYQhAXrtdqsXSt2ZahnUZ2XCh3ooFogzY"

    df = pd.read_csv(url1, parse_dates = ['supplier_price_date_updated','product_price_date_updated'])
    #df_data.loc[df_data['sale_tag']==0,'promo'] =df_data.loc[df_data['sale_tag']==0,'srp']
    df = df[['product_id', 'make','model', 'section_width', 'aspect_ratio', 'rim_size' ,'pattern', 
             'load_rating','speed_rating','stock','name','cost','srp', 'promo', 'mp_price',
             'b2b_price' , 'supplier_price_date_updated','product_price_date_updated',
             'supplier_id','sale_tag', 'promo_tag']]
    
    df = df.rename(columns={'model': 'sku_name',
                            'name': 'supplier',
                            'pattern' : 'name',
                            'make' : 'brand',
                            'section_width':'width', 
                            'rim_size':'diameter', 
                            'promo' : 'price_gulong'}).reset_index(drop = True)
 
    
    
    #df.loc[:, 'raw_specs'] = df.apply(lambda x: raw_specs(x), axis=1)
    df.loc[df['sale_tag']==0, 'price_gulong'] = df.loc[df['sale_tag']==0, 'srp']
    df.loc[:, 'width'] = df.apply(lambda x: config.clean_width(x['width']), axis=1)
    df.loc[:, 'aspect_ratio'] = df.apply(lambda x: config.clean_aspect_ratio(x['aspect_ratio']), axis=1)    
    df.loc[:, 'diameter'] = df.apply(lambda x: config.clean_diameter(x['diameter']), axis=1)
    df.loc[:, 'raw_specs'] = df.apply(lambda x: config.combine_specs(x['width'], x['aspect_ratio'], x['diameter'], mode = 'SKU'), axis=1)
    df.loc[:, 'correct_specs'] = df.apply(lambda x: config.combine_specs(x['width'], x['aspect_ratio'], x['diameter'], mode = 'MATCH'), axis=1)
    df.loc[:, 'overall_diameter'] = df.apply(lambda x: config.calc_overall_diameter(x['correct_specs']), axis=1)
    df.loc[:, 'name'] = df.apply(lambda x: config.fix_names(x['name']), axis=1)
    df.loc[:, 'sku_name'] = df.apply(lambda x: config.combine_sku(str(x['brand']), 
                                                           str(x['width']),
                                                           str(x['aspect_ratio']),
                                                           str(x['diameter']),
                                                           str(x['name']), 
                                                           str(x['load_rating']), 
                                                           str(x['speed_rating'])), 
                                                           axis=1)
    df.loc[:, 'base_GP'] = (df.loc[:, 'price_gulong'] - df.loc[:, 'cost']).round(2)
    df.loc[:, 'base_GP%'] = df.loc[:, 'base_GP']*100/df.loc[:,'price_gulong']
    active_promos = pd.read_csv('http://app.redash.licagroup.ph/api/queries/186/results.csv?api_key=8cDSJOq1Vwsc51HdjvAVQP1eQJePT5toNhFQVyzY',
                                parse_dates = ['promo_start', 'promo_end']).fillna('')
    
    #df.loc[:, 'promo_GP'] = df.apply(lambda x: config.promo_GP(x['price_gulong'], x['cost'], x['sale_tag'], x['promo_tag']), axis=1)
    df[['promo_GP', 'promo_GP%', 'promo_id']] = df.apply(lambda x: config.promo_GP(x, active_promos), axis=1, result_type = 'expand')
    df = df[df.name !='-']
    df.sort_values('product_price_date_updated', ascending = False, inplace = True)
    df.drop_duplicates(subset = ['product_id', 'sku_name', 'cost', 'price_gulong', 'supplier'])
    
    return df


@st.cache_data
def get_tire_compatible_data():
    '''

    Returns
    -------
    df : TYPE
        Gulong compatible tire sizes given car model

    '''
    
    ## retrieve acct details
    acct = bq_functions.get_acct()
    
    ## create client and credentials object
    client, credentials = bq_functions.authenticate_bq(acct)
    
    ## query dataframe
    df = bq_functions.query_bq('absolute-gantry-363408.gulong.product_tire_compatible', client)
    
    return df

def tire_select(df_data):
    '''
    Displays retention info of selected customers.

    Parameters
    ----------
    df_data : dataframe
    df_retention : dataframe
    models : list
        list of fitted Pareto/NBD and Gamma Gamma function

    Returns
    -------
    df_retention : dataframe
        df_retention with updated values

    '''
    # Reprocess dataframe entries to be displayed
    df_merged = df_data.copy()
    
    # table settings
    df_display = df_merged.sort_values(['promo_GP', 'base_GP', 'price_gulong'])
    gb = GridOptionsBuilder.from_dataframe(df_display)
    gb.configure_selection('single', 
                           use_checkbox=True, 
                           groupSelectsChildren="Group checkbox select children",
                           pre_selected_rows = [0]) #Enable multi-row selection
    gb.configure_columns(df_display.columns, width = 100)
    gb.configure_column('sku_name', 
                        headerCheckboxSelection = True,
                        width = 400)
    gridOptions = gb.build()
    
    # selection settings
    data_selection = AgGrid(
        df_display,
        gridOptions=gridOptions,
        data_return_mode='AS_INPUT', 
        update_mode='MODEL_CHANGED', 
        fit_columns_on_grid_load=False,
        enable_enterprise_modules=True,
        height= min(33*len(df_display), 400), 
        reload_data=False)
    
    selected = data_selection['selected_rows']
    
    if selected:           
        # row/s are selected
        
        df_selected = [df_display[df_display.sku_name == selected[checked_items]['sku_name']]
                             for checked_items in range(len(selected))]
        
        df_list = pd.concat(df_selected)
        #st.dataframe(df_list)    

    else:
        st.write('Click on an entry in the table to display customer data.')
        df_list = None
        
    return df_list

def compare_load_rating(ref, val):
    if pd.isna(ref) or (ref is None):
        return np.nan
    else:
        ref_list = str(ref).split('/')
        val_list = str(val).split('/')
        # both values are all numeric
        if all(r.isnumeric() for r in ref_list) and all(v.isnumeric() for v in val_list):
            if any((int(ref_list[-1])-1) <= int(v) <= (int(ref_list[0])+1) for v in val_list):
                return True
            else:
                return False
        # elif all(r.isalpha() for r in ref_list) or all(v.isalpha() for v in val_list):
        #     return np.nan
        else:
            return np.nan

def get_compatible(df, 
                   selected_row, 
                   price_range, 
                   with_load_rating = False) -> pd.DataFrame:
    
    df_temp = df.copy()
    
    price = selected_row['price_gulong'].values[0]
    load_rating = selected_row['load_rating'].values[0]
    OD = selected_row['overall_diameter'].values[0]
    df_temp.loc[:, 'od_diff'] = df_temp.overall_diameter.apply(lambda x: round(abs((x - OD)*100/OD), 2))
    
    no_matches_id = [6249, 6250, 6257, 6258, 6259,  6260]
    if ~selected_row.product_id.isin(no_matches_id):
        # 1 : lower price, higher GP, any dims, sorted by od_diff
        compatible_1 = df_temp[~df_temp.index.isin([selected_row.index]) & \
                               df_temp.od_diff.between(0, 3) & \
                               df_temp.price_gulong.between(price*(1.0-price_range/100.0), 
                                                            price) & \
                               (df_temp.promo_GP >= selected_row.promo_GP.values[0])].sort_values('od_diff', 
                                                                                        ascending = True)
        
        # 2 : any price, higher GP, any dims, sorted by od_diff
        compatible_2 = df_temp[~df_temp.index.isin([selected_row.index]) & \
                               df_temp.od_diff.between(0, 3) & \
                               df_temp.price_gulong.between(price*(1.0-price_range*0.5/100.0), 
                                                            price*(1.0+price_range*0.5/100.0))].sort_values('od_diff', 
                                                                                ascending = True)
        
        # 3 : lower price, higher GP, diff dims, sorted by od_diff
        compatible_3 = df_temp[~df_temp.index.isin([selected_row.index]) & \
                               df_temp.od_diff.between(0.01, 3) & \
                               df_temp.price_gulong.between(price*(1.0-price_range/100.0), 
                                                            price)].sort_values('od_diff', 
                                                                                ascending = True)                                                                        
         
         # 4 : any price, higher GP, diff dims, sorted by od_diff    
        compatible_4 = df_temp[~df_temp.index.isin([selected_row.index]) & \
                                df_temp.od_diff.between(0.01, 3) & \
                                df_temp.price_gulong.between(price*(1.0-price_range*0.5/100.0), 
                                                             price*(1.0+price_range*0.5/100.0))].sort_values('od_diff', 
                                                                                 ascending = True)                                                                        
        
        compatible = pd.concat([compatible_1, 
                                compatible_2, 
                                compatible_3, 
                                compatible_4], 
                               axis=0)
                                                                                                     
        if with_load_rating:
            try:
                compatible = compatible[compatible.load_rating.apply(lambda x: compare_load_rating(load_rating, x))]
            
            except:
                pass
    else:
        compatible = df_temp[df_temp.product_id.isin(no_matches_id) & \
                             (df_temp.product_id != selected_row.product_id.values[0])]
    
    # sort and cleanup
    compatible = compatible.sort_values(['promo_GP', 'price_gulong', 'od_diff'],
                        ascending = [False, True, True])\
                        .drop_duplicates(subset = 'product_id',
                                         keep = 'first')
    
    return compatible.head(5)

def optimal_price_range(df : pd.DataFrame, 
                        test_size : int = None, 
                        price_ranges : list = None,
                        with_load_rating : bool = True) -> [dict, dict]:
    # time process
    start_time = time.time()
    # preserve original dataframe & remove duplicates
    df_temp = df.copy().sort_values('promo_GP', ascending = False)\
        .drop_duplicates(subset = 'product_id', keep = 'first')
    
    # use smaller sample set if need to test function
    if test_size is not None:
        df_iter = df_temp.copy()
    else:
        df_iter = df_temp.head(test_size)
    
    # price ranges (in percent) to process
    if price_ranges is not None:
        pass
    else:
        price_ranges = [35, 40, 45, 50]

    
    # iterate over all rows
    recommended_stats = {}
    for ndx, row in df_iter.iterrows():
        recommended_stats[row.to_frame().T['sku_name'].values[0]] = {}
        for price_range in price_ranges:
            print ('Calculating {}% price range for product {}/{}'.format(price_range,
                                                                          df_iter.index.get_loc(ndx)+1, 
                                                                          len(df_iter)))
            # calculate compatible recommended items
            compatible = get_compatible(df_temp, row.to_frame().T, 
                                        price_range, with_load_rating)
            # store number of compatible items
            recommended_stats[row.to_frame().T['sku_name'].values[0]][f'{price_range}%'] = len(compatible)
    
    # calcualate price range stats
    price_range_stats = {}
    for price_range in price_ranges:
        num_series = pd.Series([recommended_stats[sku][f'{price_range}%'] for sku in recommended_stats.keys()])
        price_range_stats[f'{price_range}%'] = {'stats_summary': num_series.describe(),
                                                'value_counts' : num_series.value_counts().sort_index()}
            
    print ('Total run time: {:.2f} s'.format(time.time() - start_time))
    return price_range_stats, recommended_stats

if __name__ == '__main__':
    
    st.title('Gulong Recommendation Model')
    
    df = get_gulong_data()
    
    car_comp = get_tire_compatible_data()
    
    display_cols = ['sku_name', 'width', 'aspect_ratio', 'diameter',
                    'load_rating', 'speed_rating', 'overall_diameter', 'cost', 
                    'srp', 'price_gulong', 'mp_price', 'b2b_price', 'base_GP',
                    'promo_GP', 'promo_id']
    
    with st.sidebar:
        st.header('Tire Selection')
        
        #tab_size, tab_car = st.tabs(['By Size', 'By Car Model'])
        chosen_tab = stx.tab_bar(data = [
            stx.TabBarItemData(id = 'by_size', title = 'By Size', description = 'Filter by Tire Size'),
            stx.TabBarItemData(id = 'by_car_model', title = 'By Car Model', description = 'Filter by Car Model')
            ])
        
        placeholder = st.sidebar.container()
        # set default tab
        chosen_tab = 'by_size'
        
        if chosen_tab == 'by_size':
            
            w_list = ['Any Width'] + list(sorted(df.width.unique()))
            
            width = placeholder.selectbox('Width',
                                 options = w_list,
                                 index = 0)
            if width == 'Any Width':
                w_filter = df.copy()
            else:
                w_filter = df[df['width'] == width]
            
            
            ar_list = ['Any Aspect Ratio'] + list(sorted(w_filter.aspect_ratio.unique()))
            
            aspect_ratio = placeholder.selectbox('Aspect Ratio',
                                 options = ar_list,
                                 index = 0)
            
            if aspect_ratio == 'Any Aspect Ratio':
                ar_filter = w_filter
            else:
                ar_filter = w_filter[w_filter['aspect_ratio'] == aspect_ratio]
            
            d_list = ['Any Rim Diameter'] + list(sorted(ar_filter.diameter.unique()))
            rim_diameter = placeholder.selectbox('Rim Diameter',
                                 options = d_list,
                                 index = 0)
            
            if rim_diameter == 'Any Rim Diameter':
                final_filter = ar_filter
            else:
                final_filter = ar_filter[ar_filter['diameter'] == rim_diameter]
                
        elif chosen_tab == 'by_car_model':
            
            make_list = ['Any make'] + list(sorted(car_comp.car_make.unique()))
            
            make = placeholder.selectbox('Make',
                                 options = make_list,
                                 index = 0)
            if make == 'Any make':
                make_filter = car_comp
            else:
                make_filter = car_comp[car_comp['car_make'] == make]
            
            
            model_list = ['Any Model'] + list(sorted(make_filter.car_model.value_counts().keys()))
            
            model = placeholder.selectbox('Model',
                                 options = model_list,
                                 index = 0)
            
            if model == 'Any Model':
                model_filter = make_filter
            else:
                model_filter = make_filter[make_filter['car_model'] == model]
            
            y_list = ['Any Year'] + list(sorted(model_filter.car_year.unique()))
            year = placeholder.selectbox('Year',
                                 options = y_list,
                                 index = 0)
            
            if year == 'Any Year':
                y_filter = model_filter
            else:
                y_filter = model_filter[model_filter['car_year'] == year]
            
            final_filter = df[df.correct_specs.isin(y_filter.correct_specs.unique())]
            
        else:
            final_filter = df.copy()
            placeholder.empty()
            
    ## main window
    ## table showing tire products filtered/compatible with selected car model
    tire_selected = tire_select(final_filter[display_cols])
    
    ## calculate overall diameter
    avg_OD = final_filter['overall_diameter'].mean()
    df_temp_ = df.copy()
    df_temp_.loc[:, 'od_diff'] = df_temp_.overall_diameter.apply(lambda x: round(abs((x - avg_OD)*100/avg_OD), 2))
    
    
    with st.expander('Filtered Gulong Recommendations', expanded = True):
        st.info('Recommended tires are within 3% of AVERAGE overall diameter of tires in current selection during/after filter by tire sizes or car model')
        df_temp_ = df_temp_[df_temp_.od_diff.between(0.01, 3) & 
                            ~df_temp_.index.isin(list(final_filter.index))]\
                    .drop_duplicates(subset = 'sku_name', keep = 'first')
        st.dataframe(df_temp_[display_cols + ['od_diff']].sort_values(['od_diff', 'base_GP', 'promo_GP'],
                                                                        ascending = [True, False, False]))
    
    if tire_selected is not None:
        ## find products with overall diameter within 3% error
        OD = tire_selected.overall_diameter.unique()[0]
        # calculate overall diameter % diff
        df_temp = df.copy()
        df_temp.loc[:, 'od_diff'] = df_temp.overall_diameter.apply(lambda x: round(abs((x - OD)*100/OD), 2))
        
        price_range = st.number_input('% difference of price from selected tire',
                                      min_value = 0.0, 
                                      max_value = 100.0,
                                      value = 25.0, 
                                      step = 1.0)

        compatible = get_compatible(df_temp, tire_selected, price_range = price_range,
                                    with_load_rating = False)
        
        ## check if compatible is not empty
        if len(compatible):
            compatible = compatible.drop_duplicates(subset = 'sku_name', keep = 'first')
            
        with st.expander('**Selected Gulong Recommendations**', 
                         expanded = len(compatible)):
            
            st.info("""Recommended tires are those within ~3% change of selected tire's overall diameter.
                    Resulting recommended tires are then filtered by atleast selected tire's GP,
                    and then finally sorted by percent diff in overall diameter.""")
            
            if len(tire_selected) < len(final_filter):
                if len(compatible) == 0:
                    st.error('No recommended tires found.')
                else:
                    st.dataframe(compatible[display_cols + ['od_diff']].sort_values(['price_gulong', 'promo_GP','od_diff'],
                                                                                    ascending = [True, False, True]))
            else:
                pass
    else:
        pass
