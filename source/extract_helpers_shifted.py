
import numpy as np
import pandas as pd

#  CPs have different lengths, some dont have gbl
#  types of critical points - might have to one-hot encode
#  NACP = Nuclear Attractor Critical Point
#  NNACP = Non-Nuclear Attractor Critical Point
#  BCP = Bond Critical Point
#  RCP = Ring Critical Point
#  CCP = Cage Critical Point

# each critical point has:
# ---------------------------------------------
#   Rho = Electron Density
#   DelSqRho = Laplacian of Rho = Trace of Hessian of Rho
#   DelSqV = Laplacian of V
#   ask matthew about all the other types of laplacians
#   HessRho_EigVals = Eigenvalues of the Hessian of Rho, Ascending Order
#   Bond Ellipticity = (HessRho_EigVal(1) / HessRho_EigVal(2)) - 1
#   G = Lagrangian Form of Kinetic Energy Density
#   K = Hamiltonian Form of Kinetic Energy Density
#   L = K - G = Lagrangian Density = (-1/4)DelSqRho

#   Stress_EigVals = Eigenvalues of Stress Tensor, Ascending Order
#   V = Virial Field = Potential Energy Density = Trace of Stress Tensor
#  -DivStress = Ehrenfest Force Density = Minus Divergence of Stress Tensor


# num is an array of the critical nulcear atoms
def extract_bond_crit(num, filename="../sum_files/reactC_endo10MVc.sum"):
    lookup_other = [
        "Rho",
        "GradRho",
        "HessRho_EigVals",
        "DelSqRho",
        "V",
        "G",
        "K",
        "L",
        "Vnuc",
        "DelSqV",
        "Stress_EigVals",
        "-DivStress",
        "ESP",
        "ESPe",
        "ESPn"
    ]

    #num = sorted(num)
    iter = 0
    iter_lookup = 0
    control = 0
    ret_list = {}
    cp_of_interest = 6
    with open(filename) as myFile:
        for line_num, line in enumerate(myFile.readlines()):
            try:
                if (control == 1):

                    if (iter_lookup == len(lookup_other) or line.split() == []):
                        iter_lookup = 0
                        control = 0
                        iter += 1

                    #if (line.split()[0] == "Type"):
                    #    ret_list.append(line.split()[3])

                    if (lookup_other[iter_lookup] == line.split()[0]):

                        if (line.split()[0] == "Stress_EigVals"):
                            ret_list["Stress_EigVals_a_" + str(cp_of_interest)] = float(line.split()[2])
                            ret_list["Stress_EigVals_b_" + str(cp_of_interest)] = float(line.split()[3])
                            ret_list["Stress_EigVals_c_" + str(cp_of_interest)] = float(line.split()[4])

                        elif (line.split()[0] == "GradRho"):
                            ret_list["GradRho_a_"+ str(cp_of_interest) ] = float(line.split()[2])
                            ret_list["GradRho_b_"+ str(cp_of_interest) ] = float(line.split()[3])
                            ret_list["GradRho_c_"+ str(cp_of_interest) ] = float(line.split()[4])

                        elif (line.split()[0] == "Bond"):
                            if (line.split()[3] == "NA"):
                                ret_list["Bond_" + str(cp_of_interest)] = 0
                            else:
                                ret_list["Bond_" + str(cp_of_interest)] = float(line.split()[3])

                        elif (line.split()[0] == "HessRho_EigVals"):
                            ret_list["HessRho_EigVals_a_"+ str(cp_of_interest)] = float(line.split()[2])
                            ret_list["HessRho_EigVals_b_"+ str(cp_of_interest)] = float(line.split()[3])
                            ret_list["HessRho_EigVals_c_"+ str(cp_of_interest)] = float(line.split()[4])
                        else:
                            ret_list[lookup_other[iter_lookup] + "_" + str(cp_of_interest)] = float(line.split()[2])
                        iter_lookup += 1

                if (int(line.split()[1]) in num and line.split()[0] == "CP#"):
                    control = 1
                    cp_of_interest += 1

            except:
                pass
    return ret_list

def extract_ring_crit(num, filename="../sum_files/reactC_endo10MVc.sum"):
    lookup_other = [
        "Rho",
        "GradRho",
        "DelSqRho",
        "V",
        "G",
        "K",
        "L",
        "Vnuc",
        "DelSqV",
        "-DivStress",
        "ESP",
        "ESPe",
        "ESPn"
    ]

    iter = 0
    iter_lookup = 0
    control = 0
    ret_list = {}
    cp_of_interest = 10

    with open(filename) as myFile:
        for line_num, line in enumerate(myFile.readlines()):

            try:
                if (control == 1):

                    if (iter_lookup == len(lookup_other) or line.split() == []):
                        iter_lookup = 0
                        control = 0
                        iter += 1

                    if (lookup_other[iter_lookup] == line.split()[0]):

                        if (line.split()[0] == "Stress_EigVals"):
                            ret_list["Stress_EigVals_a_"+ str(cp_of_interest)] = float(line.split()[2])
                            ret_list["Stress_EigVals_b_"+ str(cp_of_interest)] = float(line.split()[3])
                            ret_list["Stress_EigVals_c_"+ str(cp_of_interest)] = float(line.split()[4])

                        elif (line.split()[0] == "GradRho"):
                            ret_list["GradRho_a_"+ str(cp_of_interest)] = float(line.split()[2])
                            ret_list["GradRho_b_"+ str(cp_of_interest)] = float(line.split()[3])
                            ret_list["GradRho_c_"+ str(cp_of_interest)] = float(line.split()[4])
                        elif (line.split()[0] == "HessRho_EigVals"):
                            ret_list["HessRho_EigVals_a_"+ str(cp_of_interest)] = float(line.split()[2])
                            ret_list["HessRho_EigVals_b_"+ str(cp_of_interest)] = float(line.split()[3])
                            ret_list["HessRho_EigVals_c_"+ str(cp_of_interest)] = float(line.split()[4])

                        elif (line.split()[0] == "Bond"):
                            if(line.split()[3] == "NA"):
                                ret_list["Bond_"+ str(cp_of_interest)] = 0
                            else:
                                ret_list["Bond_"+ str(cp_of_interest)] = float(line.split()[3])
                        else:
                            ret_list[lookup_other[iter_lookup]+ "_" +str(cp_of_interest)] = float(line.split()[2])

                        iter_lookup += 1

                if (int(line.split()[1]) in num and line.split()[0] == "CP#"):
                    control = 1
                    cp_of_interest += 1
            except:
                pass

    #make this a duplicate term
    if (num[0] == num[1]):
        for key in ret_list.copy():
            ret_list[key[0:-3]+"_12"] = ret_list[key]

    return ret_list

# For all nuclear critical points, there is
# i think this all are in the other part
# (1) Bader charge, - the other
# (2) energy, - in the other
# (3) electrostatic potential,
# (4) nuclear electrostatic potential,
# ---------------------------------------------
#  Vnuc = Electrostatic Potential from Nuclei
#  ESP = Total Electrostatic Potential
#  ESPe = Electrostatic Potential from Electrons
#  ESPn = Electrostatic Potential from Nuclei

# (5) multipoles,
# (6) force tensors,
# (7) spin populations - Done
# ---------------------------------------------
# Atomic Electronic Spin Populations:
# ...


def extract_nuc_crit(num, filename="../sum_files/reactC_endo10MVc.sum"):
    lookup_nuclear = [
        "Rho",
        "DelSqRho",
        "V",
        "G",
        "K",
        "L",
        "Vnuc",
        "DelSqV",
        "-DivStress",
        "ESP",
        "ESPe",
        "ESPn"
    ]

    #num = sorted(num)
    ret_list = {}
    iter = 0
    iter_lookup = 0
    control = 0
    cp_of_interest = 0

    with open(filename) as myFile:
        for line_num, line in enumerate(myFile.readlines()):
            try:
                if (control == 1):
                    if (iter_lookup == len(lookup_nuclear) or line.split() == []):
                        iter_lookup = 0
                        control = 0
                        iter += 1

                    if (line.split()[0] in lookup_nuclear[iter_lookup]  ):
                        if (line.split()[0] == "Stress_EigVals"):
                            ret_list["Stress_EigVals_a_" + str(cp_of_interest)] = float(line.split()[2])
                            ret_list["Stress_EigVals_b_" + str(cp_of_interest)] = float(line.split()[3])
                            ret_list["Stress_EigVals_c_" + str(cp_of_interest)] = float(line.split()[4])

                        elif (line.split()[0] == "GradRho"):
                            ret_list["GradRho_a_"+ str(cp_of_interest)] = float(line.split()[2])
                            ret_list["GradRho_b_"+ str(cp_of_interest)] = float(line.split()[3])
                            ret_list["GradRho_c_"+ str(cp_of_interest)] = float(line.split()[4])

                        elif (line.split()[0] == "HessRho_EigVals"):
                            ret_list["HessRho_EigVals_a_"+ str(cp_of_interest)] = float(line.split()[2])
                            ret_list["HessRho_EigVals_b_"+ str(cp_of_interest)] = float(line.split()[3])
                            ret_list["HessRho_EigVals_c_"+ str(cp_of_interest)] = float(line.split()[4])

                        else:
                            ret_list[lookup_nuclear[iter_lookup] +"_"+ str(cp_of_interest)] = float(line.split()[2])

                        iter_lookup += 1

                if (int(line.split()[1]) in num and
                        line.split()[0] == "CP#"):
                    control = 1
                    cp_of_interest += 1
            except:
                pass
        return ret_list


def extract_basics(num, filename="../sum_files/reactC_endo10MVc.sum"):

    control_1 = 0
    control_2 = 0
    iter_1 = 1
    iter_2 = 1
    ret_list = {}

    with open(filename) as myFile:
        for line_num, line in enumerate(myFile.readlines()):
            try:

                # sets control to 1 when the line of interest is hit
                if (line.split()[1] == "Atomic" and
                        line.split()[0] == "Some"):
                    control_1 = 1

                if (line.split()[1] == "Charges" and
                        line.split()[0] == "Nuclear"):
                    control_2 = 1

                # grabs charge, lagrangian, kinetic energy of ea. atom
                #if (int(line.split().count(num[iter_1 % len(num)])) > 0 and control_1 > 0):
                if (control_1 > 0):
                    ret_list["NetCharge_basic_" + str(iter_1)] = float(line.split()[1])
                    ret_list["Lagr_basic_" + str(iter_1)] = float(line.split()[2])
                    ret_list["Kinetic_basic_" + str(iter_1)] = float(line.split()[3])
                    ret_list["K|Scaled|_basic_" + str(iter_1)] = float(line.split()[4])
                    iter_1 += 1

                # grabs position of six atoms
                if (control_2 > 0):
                    ret_list["charge_basic_"+str(iter_2)] = float(line.split()[2])
                    ret_list["x_basic_"+str(iter_2)] = float(line.split()[3])
                    ret_list["y_basic_"+str(iter_2)] = float(line.split()[4])
                    ret_list["z_basic_"+str(iter_2)] = float(line.split()[5])
                    iter_2 += 1

                if (iter_1-1 >= len(num)):
                    control_1 = 0
                    iter_1 = 1
                if (iter_2-1 >= len(num)):
                    control_2 = 0
                    iter_2 = 1

            except:
                pass
        return ret_list

# this implementation gets "N_total" and "N_spin" vectors from the
# atoms in the cage of the diels alder rxn
def extract_charge_energies(num, filename="../sum_files/reactC_endo10MVc.sum"):

    with open(filename) as myFile:
        control = 0
        iter = 1
        ret_list = {}
        for line_num, line in enumerate(myFile.readlines()):
            try:
                if (line.split()[0] == "Molecular" and line.split()[1] == "energy"):
                    ret_list["MolEnergy"] = float(line.split()[-1])
                    
                if (line.split()[0] == "Some" and line.split()[1] == "Atomic"
                        and line.split()[2] == "Properties:"):
                    control = 1

                if ((line.split()[0].lower() in num or line.split()[0].upper() in num) and control == 1):
                    ret_list["NetCharge_" + str(iter)] = float(line.split()[1])
                    ret_list["Lagrangian_" + str(iter)] = float(line.split()[2])
                    iter += 1

                if (iter-1 >= len(num)):
                    control = 0

            except:
                pass

    return ret_list

# this implementation gets "N_total" and "N_spin" vectors from the
# atoms in the cage of the diels alder rxn
def extract_spin(num, filename="../sum_files/reactC_endo10MVc.sum"):
    with open(filename) as myFile:

        control = 0
        ret_dic = {}
        iter = 1

        for line_num, line in enumerate(myFile.readlines()):
            try:

                if (line.split()[0] == "Atomic" and line.split()[1] == "Electronic"
                        and line.split()[2] == "Spin"):
                    control = 1

                if ((line.split()[0].lower() in num or line.split()[0].upper() in num) and control == 1):

                    ret_dic["Spin_tot_" + str(iter)] = float(line.split()[3])
                    ret_dic["Spin_net_" + str(iter)] = float(line.split()[4])
                    iter += 1

                if (iter-1 >= len(num)):
                    control = 0

            except:
                pass

    return ret_dic

def atom_xyz_from_sum(filename=""):
    temp = []
    xyz = []
    control = 0

    with open(filename) as myFile:
        for line_num, line in enumerate(myFile.readlines()):
            try:
                if (line.split() == []):
                    control = 0

                if (line.split()[0] == 'Atom' and line.split()[1] == 'Charge'
                        and line.split()[2] == 'X'):
                    control = 1
                    print("found atom xyz in file")

                tf = line.split()[0][0] == 'H' or line.split()[0][0] == 'N' or line.split()[0][0] == 'C' or \
                     line.split()[0][0] == 'O' or \
                     line.split()[0][0] == 'h' or line.split()[0][0] == 'n' or line.split()[0][0] == 'c' or \
                     line.split()[0][0] == 'o' or line.split()[0] == 'B' or line.split()[0] == 'F' \
                     or line.split()[0] == 'O' or line.split()[0] == 'Si' or line.split()[0] == 'si' \
                     or line.split()[0] == 'SI'


                if (control == 1 and tf):

                    temp.append(line.split()[-5][0])
                    temp.append(float(line.split()[-3]))
                    temp.append(float(line.split()[-2]))
                    temp.append(float(line.split()[-1]))
                    print(temp[0], temp[1], temp[2], temp[3])
                    xyz.append(temp)
                    temp = []

            except:
                pass

def extract_all():
    # add reactD once I get the files for it
    fl_arr = []
    y = []
    list_of_dicts = []
    df = pd.read_excel("../data/barriers.xlsx")
    full_dictionary = {}

    for i, j in enumerate(df["group"]):
        if (j != "reactC"):
            temp_name = "../sum_files/" + str(j) + "_" + str(df["file"].values[i]) + "-opt.sum"
            fl_arr.append(temp_name)

        else:
            temp_name = "../sum_files/" + str(j) + "_" + str(df["file"].values[i]) + ".sum"
            fl_arr.append(temp_name)

    # extracts xyz position of the first diels-alder atom in the first file. Used to standardize position
    atoms = df["AtomOrder"][0]
    atoms = atoms.replace(" ", "")[1:-1]
    atom_id = [x[1:-1] for x in atoms.split(",")]
    basis = extract_basics(num=atom_id, filename=fl_arr[0])
    basis_atom_1 = [basis["x_basic_1"], basis["y_basic_1"], basis["z_basic_1"]]

    for ind, fl in enumerate(fl_arr):
        # if (pd.isnull(df["AtomOrder"][ind]) or pd.isnull(df["CPOrder"][ind]) ):
        #    #print("critical points not added here")
        #    pass
        # else:
        atoms = df["AtomOrder"][ind]
        atoms = atoms.replace(" ", "")[1:-1]
        atom_id = [x[1:-1] for x in atoms.split(",")]
        bond_cp = [int(x) for x in df["CPOrder"][ind].split(",")][6:13]
        ring_cp = bond_cp[4:6]
        bond_cp = bond_cp[0:4]

        atoms = []
        for i in atom_id:
            if (len(i) == 3):
                atoms.append(int(i[1:3]))
            else:
                atoms.append(int(i[-1]))

        bond = extract_bond_crit(num=bond_cp, filename=fl)
        ring = extract_ring_crit(num=ring_cp, filename=fl)
        nuc = extract_nuc_crit(num=atoms, filename=fl)
        charge = extract_charge_energies(num=atom_id, filename=fl)
        spin = extract_spin(num=atom_id, filename=fl)
        basics = extract_basics(num=atom_id, filename=fl)

        translate = ["x_basic_1", "y_basic_1", "z_basic_1",
                     "x_basic_2", "y_basic_2", "z_basic_2", "x_basic_3", "y_basic_3", "z_basic_3",
                     "x_basic_4", "y_basic_4", "z_basic_5", "x_basic_5", "y_basic_5", "z_basic_5",
                     "x_basic_6", "y_basic_6", "z_basic_6"]

        for i in translate:

            if (i[0] == 'x'):
                basics[i] = basics[i] - basis_atom_1[0]
            elif (i[0] == 'y'):
                basics[i] = basics[i] - basis_atom_1[1]
            else:
                basics[i] = basics[i] - basis_atom_1[2]

        # print(len(bond) + len(ring) + len(nuc) + len(charge)+\
        #      len(spin) + len(basics))
        full_dictionary.update(bond)
        full_dictionary.update(ring)
        full_dictionary.update(nuc)
        full_dictionary.update(charge)
        full_dictionary.update(spin)
        full_dictionary.update(basics)


        y.append(float(df["barrier(kj/mol)"][ind]))

        list_of_dicts.append(full_dictionary)
        full_dictionary = {}

    df_results = pd.DataFrame(list_of_dicts)

    return df_results, np.array(y)

# from extract_helpers import atom_xyz_from_sum
# atom_xyz_from_sum("../sum_files/A_1-opt.sum")
