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

def extract_other_crit(num, filename="../sum_files/reactC_endo10MVc.sum"):
    lookup_other = [
        "Rho",
        "GradRho",
        "HessRho_EigVals",
        "DelSqRho",
        "Bond",
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
        "ESPn",
        "GBL"
    ]

    num = sorted(num)
    iter = 0
    iter_lookup = 0
    control = 0
    ret_list = {}
    list_dicts = []

    with open(filename) as myFile:
        for line_num, line in enumerate(myFile.readlines()):

            try:
                if (control == 1):

                    if (iter_lookup == len(lookup_other) or line.split() == []):
                        iter_lookup = 0
                        control = 0
                        iter += 1
                        list_dicts.append(ret_list)
                        ret_list = {}

                    if (line.split()[0] == "Type"):
                        ret_list.append(line.split()[3])

                    if (lookup_other[iter_lookup] == line.split()[0]):

                        if (line.split()[0] == "Stress_EigVals"):
                            ret_list["Stress_EigVals"] = line.split()[2:5]
                        elif (line.split()[0] == "GradRho"):
                            ret_list["GradRho"] = line.split()[2:5]

                        elif (line.split()[0] == "Bond"):
                            ret_list["Bond"] = line.split()[3]

                        elif (line.split()[0] == "HessRho_EigVals"):
                            ret_list["HessRho_EigVals"] = line.split()[2:5]
                        else:
                            ret_list[lookup_other[iter_lookup]] = line.split()[2]

                        iter_lookup += 1

                if (int(line.split()[1]) == num[iter] and
                        line.split()[0] == "CP#"):
                    control = 1

            except:
                pass
    return list_dicts


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
def extract_nuc_crit(num, filename="../sum_files/reactC_endo10MVc.sum"):
    lookup_nuclear = [
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
        "ESPn",
    ]

    num = sorted(num)
    ret_list = {}
    list_dicts = []
    iter = 0
    iter_lookup = 0
    control = 0

    with open(filename) as myFile:
        for line_num, line in enumerate(myFile.readlines()):

            try:
                if (control == 1):

                    if (iter_lookup == len(lookup_nuclear) or line.split() == []):
                        iter_lookup = 0
                        control = 0
                        iter += 1
                        list_dicts.append(ret_list)
                        ret_list = {}

                    if (line.split()[0] == "Type"):
                        ret_list.append(line.split()[3])

                    if (lookup_nuclear[iter_lookup] == line.split()[0]):
                        if (line.split()[0] == "Stress_EigVals"):
                            ret_list["Stress_EigVals"] = line.split()[2:5]
                        elif (line.split()[0] == "GradRho"):
                            ret_list["GradRho"] = line.split()[2:5]
                        elif (line.split()[0] == "HessRho_EigVals"):
                            ret_list["HessRho_EigVals"] = line.split()[2:5]

                        else:
                            ret_list[lookup_nuclear[iter_lookup]] = line.split()[2]

                        iter_lookup += 1

                if (int(line.split()[1]) == num[iter] and
                        line.split()[0] == "CP#"):
                    control = 1

            except:
                pass
        return list_dicts


# note, might have to modulate the string traversing based on the number of atoms in the file
# bond pulling stil isn't achieved

def extract_basics(num, filename="../sum_files/reactC_endo10MVc.sum"):
    lookup_dictionary = [
        "Number of electrons",
        "Nuclear Charges and Cartesian Coordinates",
        "n_atoms",
        "L(A)",
        "Molecular energy E(Mol) from the wfn file:",
        "1st Largest",
        "Atomic Electronic Spin Populations",
        "Atomic Dipole Moments:",
    ]

    control_1 = 0
    control_2 = 0

    ret_list = []
    iter = 0

    with open(filename) as myFile:
        for line_num, line in enumerate(myFile.readlines()):
            try:

                ########
                if (iter > 2 * len(num) - 1):
                    break

                if (line.split()[1] == "Atomic" and
                        line.split()[0] == "Some"):
                    control_1 = 1

                if (line.split()[1] == "Charges" and
                        line.split()[0] == "Nuclear"):
                    control_2 = 1

                # grabs charge, lagrangian, kinetic energy of ea. atom
                if (int(line.split().count(num[iter % len(num)])) > 0 and control_1 > 0):
                    ret_list = ret_list + [float(i) for i in line.split()[1:4]]
                    iter += 1

                # grabs position of six atoms
                if (int(line.split().count(num[iter % len(num)])) > 0 and control_2 > 0):
                    ret_list = ret_list + [float(i) for i in line.split()[2:5]]
                    iter += 1

                if (iter >= len(num)):
                    control = 0

            except:
                pass
        return ret_list


# this implementation gets "N_total" and "N_spin" vectors from the
# atoms in the cage of the diels alder rxn

def extract_charge_energies(num, filename="../sum_files/reactC_endo10MVc.sum"):
    with open(filename) as myFile:
        control = 0
        iter = 0

        ret_list = []
        ret_list1 = []
        ret_list2 = []

        for line_num, line in enumerate(myFile.readlines()):
            try:
                if (line.split()[0] == "Molecular" and line.split()[1] == "energy"):
                    ret_list.append(float(line.split()[-1]))

                if (line.split()[0] == "Some" and line.split()[1] == "Atomic"
                        and line.split()[2] == "Properties:"):
                    control = 1

                if (num[iter] == line.split()[0] and control == 1):
                    ret_list1.append(float(line.split()[1]))
                    ret_list2.append(float(line.split()[2]))
                    iter += 1

                if (iter >= len(num)):
                    control = 0
            except:
                pass

    ret_list1 = ret_list + ret_list1 + ret_list2
    return ret_list1


# this implementation gets "N_total" and "N_spin" vectors from the
# atoms in the cage of the diels alder rxn

def extract_spin(num, filename="../sum_files/reactC_endo10MVc.sum"):
    with open(filename) as myFile:
        control = 0
        ret_list1 = []
        ret_list2 = []

        iter = 0
        for line_num, line in enumerate(myFile.readlines()):
            try:

                if (line.split()[0] == "Atomic" and line.split()[1] == "Electronic"
                        and line.split()[2] == "Spin"):
                    control = 1

                if (num[iter] == line.split()[0] and control == 1):
                    ret_list1.append(float(line.split()[3]))
                    ret_list2.append(float(line.split()[4]))
                    iter += 1

                if (iter >= len(num)):
                    control = 0

            except:
                pass

    ret_list1 = ret_list1 + ret_list2
    return ret_list1


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
                     line.split()[0][0] == 'o'

                line.split()[0] == 'B' or line.split()[0] == 'F' or line.split()[0] == 'O'
                if (control == 1 and tf):
                    temp.append(line.split()[-5][0])
                    temp.append(float(line.split()[-3]))
                    temp.append(float(line.split()[-2]))
                    temp.append(float(line.split()[-1]))

                    # print(temp)
                    print(temp[0], temp[1], temp[2], temp[3])
                    xyz.append(temp)

                    temp = []



            except:
                pass

# from extract_helpers import atom_xyz_from_sum
# atom_xyz_from_sum("../sum_files/A_1-opt.sum")
