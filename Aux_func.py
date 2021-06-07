from pyomo.core import Var
import pandas as pd
from argparse import ArgumentTypeError

def model_res_to_dict(m):
    '''Function to save the output of the model in a 2-levels dictionary: first
    level are the variables, second level are countries-time periods keys'''
    res_dict =  {str(v): {index: getattr(m, str(v))[index].value 
                       for index in getattr(m, str(v))}
                       for v in m.component_objects(Var, active=True)}
    return res_dict

def output_format(countries, out_unformat, t, T):
    '''Function to better formatting the model output and having it ready for
    being saved on excel files. The output is a list of lists with countries as
    first element and a DataFrame with all variables as index and time periods
    as columns. Country independent variables are grouped under the "global tag" '''
    var_l = list(out_unformat.keys())
    var_c = var_l[:-7]
    var_gl = var_l[-7:]
    out_form = {i: pd.DataFrame(data = float(0), index = var_c, columns = t) for i in countries}
    out_form['global'] = pd.DataFrame(data = float(0), index = var_gl, columns = t)    
    idx = []
    for j in var_l:
        if isinstance(list(out_unformat[j].keys())[0], tuple):
            for i in countries:
                for k in range(1,T):            
                    idx.append((i, j, k))
        else:
            for k in range(1,T):
                idx.append(('global', j, k))               
    for i in idx:
        if i[0] != 'global':                
            out_form[i[0]].at[i[1],i[2]] = out_unformat[i[1]][(i[0],i[2])]
        else:
            out_form[i[0]].at[i[1],i[2]] = out_unformat[i[1]][i[2]]
    return out_form

def results_to_excel(res, countries, results_path, filename):
    '''Function to export results on a Excel file for full coopearative and 
    non-cooperative (no coalitions) case. Each country has a worksheet with
    variables as rows and time periods as columns. A "global" worksheet
    contains country-independent variables.''' 
    final_path = results_path + filename    
    writer = pd.ExcelWriter(final_path)
    c_list = [i for i in countries]
    c_list.append('global')
    for i in c_list:
        res[i].to_excel(writer, i)
    writer.save()        
    
    
def coa_f(N):
    '''Function that, given an integer N supposed to be the cardinality of an 
       ordered set of players of a colaitioal game, gives back the power set of N, 
       or else, all the possible coalitions given N in form of a matrix (numpy.array) 
       where 1 indicates a player belonging to that coalition and 0 the contrary.
       The empty set is excluded by the list of all possible coalitions.'''
    from itertools import product, repeat
    coat = list(product(*repeat(range(2), N)))
    coa = []
    for i in range(len(coat)):
        coa.append(list(coat[i]))  
    # Sorting coalitions accordig to number of players present
    coa2 = [[] for i in range(N+1)]
    for i in range(len(coa)):
        for l in range(len(coa)):
            if sum(coa[l]) == i:
                coa2[i].append(coa[l])            
    # Sorting players in coalitions (firsts come first)
    for i in range(len(coa2)):
        coa2[i].sort(reverse=True)
    coa_x = [coa2[i][f] for i in range(1,N+1) for f in range(len(coa2[i]))]   
    return coa_x  

def check_arg_T(arg):
    '''Check that the given number of time periods is of type int and inside bounds'''
    try:
        f = int(arg)
    except ValueError:    
        raise ArgumentTypeError("Must be an integer number")
    if f < 2 or f > 59:
        raise ArgumentTypeError("Argument must be an integer < " + str(60) + "and > " + str(1))
    return f

def check_arg_tstep(arg):
    try:
        f = int(arg)
    except ValueError:    
        raise ArgumentTypeError("Must be an integer number") 
    if f not in [1, 2, 5, 10, 20]:
        raise ArgumentTypeError("Argument must be one of the following integer values: 1, 2, 5, 10, 20")
    return f

def check_arg_tol(arg):
    '''Check that the given tolerance is of type int and inside bounds'''
    try:
        f = int(arg)
    except ValueError:    
        raise ArgumentTypeError("Must be an integer number")
    if f < 7 or f > 12:
        raise ArgumentTypeError("Argument must be an integer < " + str(13) + "and > " + str(6))
    return f

def check_arg_max_iter(arg):
    '''Check that the given maximum number of iterations is of type int and inside bounds'''
    try:
        f = int(arg)
    except ValueError:    
        raise ArgumentTypeError("Must be an integer number")
    if f < 500 or f >25000:
        raise ArgumentTypeError("Argument must be an integer < " + str(25001) + "and > " + str(499))
    return f

def check_bool_arg(arg):
    '''Check that the provided argument is a string equal to True or False and return appropriate boolean'''
    if str(arg) != 'False' and str(arg) != 'True':
        raise ArgumentTypeError("--coop and --nc only accept True or False as given values!")
    else:
        return str(arg)

def coa_to_analyse(arg):
    if arg != 'none' and arg != 'all':
        all_c = {'US':0, 'EU':0, 'JAP':0, 'RUS':0, 'EUR':0, 'CHI':0, 'IND':0, 
                 'MEST':0, 'AFR':0, 'LAM':0, 'OHI':0, 'OTH':0}
        l_countries = arg.split(',')
        l_c2 = [i.replace(" ","") for i in l_countries]
        problem = 0
        for i in l_c2:
            try:
                aaa = all_c[i]
                all_c[i] = 1
            except KeyError:
                problem += 1
        if problem == 0:
            return list(all_c.values())
        else:
            raise ArgumentTypeError('You have probably inserted a wrong \
                                             string of countries-regions in the --coalition argument. \
                                             Valid countries-regions are: US, EU, JAP, RUS, EUR, CHI, IND, MEST, AFR, LAM, OHI, OTH')

