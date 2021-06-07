
# =============================================================================
# This file contains the function to perform the stability analysis of coalitions
# =============================================================================

def c_f_dif(coa, pay):
    '''This function computes the difference between the pay-off of a player when she is member of a coalition,
    and her pay-off when she is the only one to abandon the same coalition'''
    N = 12
    f_l = [[[] for i in range(N)] for j in coa]
    for j in range(len(coa)):
        if sum(coa[j]) >1:
            c_sh = coa[:j]
            for i in range(N):
                if coa[j][i]==1:
                    for k in range(len(c_sh)):
                        c1 = [coa[j][l] for l in range(N) if l!=i]
                        c2 = [c_sh[k][l] for l in range(N) if l!=i]
                        if all(m[0]==m[1] for m in zip(c1,c2)):
                            f_l[j][i] = pay[j][i] - pay[k][i]                
                else:
                    f_l[j][i] = 0    
    return f_l

def int_st(c_diff, coa):
    '''Function that, given the result of c_f_diff() and the list of coalitions, returns the list of all internally stable coalitions,
    and the list of the coalitions that are internally stable only through transfers'''
    N = 12                 
    coa = coa[N:]
    c_f_d = c_diff[N:]
    i_s = [list(coa[j]) if all(i >=-0.000001 for i in c_f_d[j]) else 0 for j in range(len(c_f_d))]
    i_s_wt = [list(coa[j]) for j in range(len(coa)) if type(i_s[j])==int and sum(c_f_d[j])>=0]
    return [i_s, i_s_wt]  

def dif_ext(coa, pay):
    '''This function computes the difference between the pay-off of a player not member of a coalition,
    and her pay-off when she is the only one to join the same coalition'''
    from numpy import subtract
    N = 12
    dif = [[0 for i in range(N)] for j in coa]
    for i in range(N+1, len(coa)):
        coab = [j for j in range(N, len(coa)) if sum(coa[j])==sum(coa[i])+1]
        fc = []
        for j in coab:
            if sum(list(subtract(coa[j],coa[i]))) == 1 and -1 not in list(subtract(coa[j],coa[i])):
                fc.append(j)
        for j in fc:
            for k in range(N):
                if coa[i][k] != coa[j][k]:
                    dif[i][k] = pay[i][k]-pay[j][k]
    return dif

def ext_st(dif_ext, coa):
    '''This function returns the list of externally stable coalitions'''
    N = 12
    ddd = dif_ext
    exs = [list(coa[j]) if all(i >=-0.000001 for i in dif_ext[j]) and j >N+1 else 0 for j in range(len(dif_ext))]
    exs_wt = [list(coa[j]) for j in range(len(coa)) if type(exs[j])==int and sum(ddd[j])>=0]    
    return [exs, exs_wt] 

def stab_c(coal, int_st, ext_st):
    '''This function returns the list of all stable coalitions (interally and externally), in form of 0-1 lists'''
    coal2 = [list(i) for i in list(coal)]
    stab = [i if i in ext_st and i in int_st else 0 for i in coal2]
    return stab

def coa_int(s_coa):
    ''' Given a coalition in form of a list of 0-1 values, it returns the list
    of coalitions (in the same form) required to evaluate the internal stability 
    of the given coalition'''
    memb_id = [i for i in range(len(s_coa)) if s_coa[i] == 1] 
    c_int = []
    for i in memb_id:
        s_mod = s_coa.copy()
        s_mod[i] = 0
        c_int.append(s_mod)
    return c_int    

def coa_ext(s_coa):
    ''' Given a coalition in form of a list of 0-1 values, it returns the list
    of coalitions (in the same form) required to evaluate the external stability 
    of the given coalition'''
    memb_id = [i for i in range(len(s_coa)) if s_coa[i] == 0] 
    c_ext = []
    for i in memb_id:
        s_mod = s_coa.copy()
        s_mod[i] = 1
        c_ext.append(s_mod)
    return c_ext    