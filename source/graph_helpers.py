import pandas as pd 
import numpy as np




def extract_nuc_crit_full(num, filename="../sum_files/reactC_endo10MVc.sum"):
    lookup_nuclear = ["DelSqRho", "V", "ESP"]

    # num = sorted(num)
    atom_count = 0 
    control = 0
    cp_of_interest = 0
    control_atom_counter = 0 
    control = 1 
    ret_list = {}
    iter = 0
    iter_lookup = 0
    
    with open(filename) as myFile:
        for line_num, line in enumerate(myFile.readlines()):

            try:
                if line.split()[0] == 'Atom' and line.split()[1] == 'Charge' and line.split()[2] == 'X':
                    control_atom_counter = 1
            
                if control_atom_counter > 0:
                    atom_count += 1 
                    if(line.split()[0] == 'q(A)' and line.split()[1] == '=' and line.split()[2] == 'Net'):
                        control_atom_counter = -1 
                        
                if control == 1:
                    if iter_lookup == len(lookup_nuclear) or line.split() == []:
                        iter_lookup = 0
                        control = 0
                        iter += 1

                    if line.split()[0] in lookup_nuclear[iter_lookup]:
                        ret_list[
                            "$\mathcal{"
                            + lookup_nuclear[iter_lookup]
                            + "}"
                            + "_{"
                            + str(cp_of_interest)
                            + "}$"
                        ] = float(line.split()[2])

                        iter_lookup += 1



                if line.split()[0] == "CP#":
                    control = 1
                    cp_of_interest += 1
            
                if(num == -1):
                    control_count = atom_count
                else: 
                    control_count = len(num)
                
                if iter >= control_count:
                    control = 0

            except:
                pass
        print(atom_count)
        return ret_list


def extract_basics(num, filename="../sum_files/reactC_endo10MVc.sum"):
    print("new function")
    control_1 = 0
    control_2 = 0
    iter_1 = 0
    iter_2 = 0
    ret_list = {}
    atom_count = -3
        
    with open(filename) as myFile:
        print(filename)
        for line_num, line in enumerate(myFile.readlines()):
            
            try:
                # sets control to 1 when the line of interest is hit
                if line.split()[0] == 'Atom' and line.split()[1] == 'Charge' and line.split()[2] == 'X':
                    control_atom_counter = 1

                if line.split()[1] == "Atomic" and line.split()[0] == "Some":
                    control_1 = 1

                if line.split()[1] == "Charges" and line.split()[0] == "Nuclear":
                    control_2 = 1

                # grabs charge, lagrangian, kinetic energy of ea. atom
                if control_atom_counter > 0:
                    atom_count += 1 
                    if(line.split()[0] == 'q(A)' and line.split()[1] == '=' and line.split()[2] == 'Net'):
                        control_atom_counter = -1 

                if control_1 > 0:
                    ret_list["Lagr_basic_" + str(iter_1)] = float(line.split()[2])
                    ret_list["Kinetic_basic_" + str(iter_1)] = float(line.split()[3])
                    iter_1 += 1

                # grabs position of six atoms
                if control_2 > 0:
                    ret_list["charge_basic_" + str(iter_2)] = float(line.split()[2])
                    ret_list["x_basic_" + str(iter_2)] = float(line.split()[3])
                    ret_list["y_basic_" + str(iter_2)] = float(line.split()[4])
                    ret_list["z_basic_" + str(iter_2)] = float(line.split()[5])
                    iter_2 += 1

                if(num == -1):
                    control_count = atom_count
                else: 
                    control_count = len(num)
                
                if iter_1 >= control_count:
                    control_1 = 0

                if iter_2 >= control_count:
                    control_2 = 0


            except:
                pass
        return ret_list




def extract_all():
    # add reactD once I get the files for it
    fl_arr = []
    y = []
    list_of_dicts = []
    df = pd.read_excel("../data/barriers.xlsx", engine = 'openpyxl')
    df = df.replace(r'^\s+$', np.nan, regex=True).dropna(how='all')
    
    full_dictionary = {}
    for i, j in enumerate(df["group"]):
        if j != "reactC":
            temp_name = (
                "../sum_files/" + str(j) + "_" + str(df["file"].values[i]) + "-opt.sum"
            )
            fl_arr.append(temp_name)

        else:
            temp_name = (
                "../sum_files/" + str(j) + "_" + str(df["file"].values[i]) + ".sum"
            )
            fl_arr.append(temp_name)
    
    for ind, fl in enumerate(fl_arr):
        nuc = extract_nuc_crit_full(num=-1, filename=fl)
        basics = extract_basics(num=-1, filename=fl)
        print(nuc)
        
        #print(basics)
        
        #full_dictionary.update(nuc)
        full_dictionary.update(basics)

        y.append(float(df["barrier(kj/mol)"][ind]))
        list_of_dicts.append(full_dictionary)
        full_dictionary = {}
    df_results = pd.DataFrame(list_of_dicts)
    return df_results, np.array(y)



df_results , y = extract_all()
