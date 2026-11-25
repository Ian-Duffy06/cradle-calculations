#%%
#-------------------------Imports--------------------------
import numpy as np
import matplotlib.pyplot as plt
from math import factorial

#------------------------------TBM Measurements and Geometry--------------------------
r_cm  = 29.619 #cm                    #radius of TBM
d_cm  = r_cm*2 #cm                    #diameter of TBM
d_in = d_cm/2.54 #in                  #diameter of TBM
r  = ((r_cm/100)*3.281*12) #in        #radius of TBM
L  = 97.8 #in                         #total length of TBM
L1 = 40.5 #in                         #length of TBM section 1 (75% of weight)
L2 = 27.3 #in                         #length of TBM section 2 (10%)
L3 = 30 #in                           #length of TBM section 3 (15%)
w_tbm_kg = 516.8 #kg                  #total weight of TBM
w_tbm_lb = w_tbm_kg*2.20462 #lb       #total weight of TBM (1137 lb)
w1 = (w_tbm_lb*.75)/L1 #lb/in         #weight per unit length, section 1
w2 = (w_tbm_lb*.10)/L2 #lb/in         #weight per unit length, section 2
w3 = (w_tbm_lb*.15)/L3 #lb/in       #weight per unit length, section 3
if abs(L1 + L2 + L3 - L) > 0.05:
    print(f"WARNING: L1+L2+L3 = {L1+L2+L3:.1f} in but L = {L:.1f} in")

#------------------------TBM Forces on the Tube--------------------------
d_opening = 6.5 #in                     #total width of the opening for the torque reaction system
h_opening = 1.5 #in                 #height of the opening for the torque reaction system
d_tube = d_opening / 2 #in             #tube contact distance from TBM centreline
u = 0.8                             #friction coefficient, TBM on steel tube; worst case friction scenario
F = w_tbm_lb #lb                    #force along the radius
theta = np.arcsin(d_tube/r) #rad    #contact angle
theta_deg = np.degrees(theta) #deg  #contact angle
h = r*np.cos(theta) #in             #height of the TBM centreline above the tube
h_outer_to_beam_top =  r - h #in              #height of the torque reaction system to height of beam
h_beam = round(h_outer_to_beam_top + h_opening, 2) #in  #height of beam required given the width of the torque reaction system opening
F_x = F*np.sin(theta) #lb           #sideways part
F_y = F*np.cos(theta) #lb           #vertical part
Friction_force = u*F #lb            #along the beam
print("The height of the beam required to hold the TBM at the given Torque system opening is:", f"{h_beam:.2f} in")
print("Force in the vertical direction:", f"{F_y:.2f} lb")
print("Force in the horizontal direction:", f"{F_x:.2f} lb")
print("Angle of contact:", f"{theta_deg:.2f} deg")
#-------------------------AISC Tube: HSS 4x2x1/8--------------------------
#Need to change if tube dimensions change
Tube = (round(h_beam, 1), 2, (1/4))           #in, (height, width, wall thickness)
w_tube = 5.41 #lb/f                 #weight of tube (4.75 lb/ft)
w_self = w_tube/12 #lb/in           #self-weight (0.3958 lb/in)
A = 1.51 #in^2                      #area per ASIC HSS tables
t = .233 #in                        #design wall thickness per AISC HSS tables
h = Tube[0] #in                     #height
b = Tube[1] #in                     #width
I_x, Z_x, S_x, r_y = .747, .964, .747, .704     #strong axis: in^4, in^3, in^3
I_y, Z_y, S_y, r_x = .747, .964, .747, .704     #weak axis:   in^4, in^3, in^3
J = 1.31 #in^4                       #torsional constant

b_t = (b - 3*t)/t                        #flange flat-width ratio
h_t = (h - 3*t)/t                        #web flat-width ratio
Q_x = (b*t)*((h-t)/2) + t*((h/2)-t)**2   #in^3, first moment for vertical shear
Q_y = (h*t)*((b-t)/2) + t*((b/2)-t)**2   #in^3, first moment for sideways shear
A_m = (b-t)*(h-t) #in^2                  #area enclosed by wall centreline (torsion)
#------------------------Material--------------------------
E  = 29000000 #psi                  #modulus of elasticity
Fy = 32000 #psi                     #yield strength (worst case)
Fy_shear = 0.6*Fy #psi              #shear AND torsional yield (AISC G and H3)
#%%
#================== BEAM ON 4 TIES: REACTIONS, SHEAR, MOMENT, DEFLECTION ==================
CASE = "point"      # "point"  = all of W as one point load on one beam (worst case)
                    # "normal" = machine spread along both beams (75/10/15 split)
PLOT = True         # "lifting" = forces that occur when lifting the TBM (vertical and horizontal)

L_beam   = 88.5                       # in, beam length
ties     = [1.0, 29.6, 59.0, 87.5]    # in, tie centres (x from front end of beam)
#ties     = [0, 1, 87.5, 88.5]
overhang = (L - L_beam)/2             # in, machine past each beam end
a_pt     = (ties[0] + ties[1])/2      # in, point load at mid first span
span_max = max(np.diff(ties))         # in, longest span between ties
xs = np.linspace(0, L_beam, 2001)

def mac(x, s, n):
    """Macaulay term <x - s>^n / n!  (n = 0 gives a step that switches on past s)."""
    x = np.asarray(x, float)
    return np.where(x > s, (x - s)**n, 0.0) / factorial(n)

def loads_term(x, point_loads, dist_loads, n):
    """Known-load part of the n-th Macaulay function (n=0: V, n=1: M, n=3: EI*y)."""
    out = sum(P*mac(x, s, n) for P, s in point_loads)
    out = out + sum(q*(mac(x, x0, n+1) - mac(x, x1, n+1)) for q, x0, x1 in dist_loads)
    return out

def solve_beam(point_loads, dist_loads, I_used):
    """Solve the 6x6 (y = 0 at 4 ties, sum F, sum M). Returns R, V, M, y on xs."""
    A_mat = [[float(mac(x, s, 3)) for s in ties] + [x, 1.0] for x in ties]
    rhs   = [float(loads_term(x, point_loads, dist_loads, 3)) for x in ties]
    A_mat += [[1, 1, 1, 1, 0, 0], ties + [0, 0]]
    rhs   += [sum(P for P, _ in point_loads) + sum(q*(x1-x0) for q, x0, x1 in dist_loads),
              sum(P*s for P, s in point_loads) + sum(q*(x1-x0)*(x0+x1)/2 for q, x0, x1 in dist_loads)]
    *R, C1, C2 = np.linalg.solve(np.array(A_mat), np.array(rhs))

    def f(n):  # reactions minus loads, n-th Macaulay function
        return sum(Ri*mac(xs, s, n) for Ri, s in zip(R, ties)) - loads_term(xs, point_loads, dist_loads, n)
    V, M = f(0), f(1)
    y = (f(3) + C1*xs + C2) / (E*I_used)
    return np.array(R), V, M, y

def peak(arr):
    i = np.argmax(abs(arr)); return abs(arr[i]), xs[i]

# ---- Vertical and horizontal loads for the chosen case ----
if CASE == "lifting":
    pass 
if CASE == "point":
    vert_pts,  vert_dist  = [(F_y, a_pt)], []
    horiz_pts, horiz_dist = [(F_x, a_pt)], []
elif CASE == "normal":
    x12 = L1 - overhang                 # end of section 1 on the beam
    x23 = x12 + L2                      # end of section 2 on the beam
    q1, q2, q3 = w1/2, w2/2, w3/2       # vertical lb/in on one beam (half the machine)
    vert_dist = [(q1, 0, x12), (q2, x12, x23), (q3, x23, L_beam)]
    vert_pts  = [(q1*overhang, 0.0), (q3*overhang, L_beam)]          # overhanging machine
    horiz_dist = [(q*np.tan(theta), x0, x1) for q, x0, x1 in vert_dist]
    horiz_pts  = [(P*np.tan(theta), x) for P, x in vert_pts]
else:
    raise ValueError('CASE must be "point" or "normal"')
vert_dist = vert_dist + [(w_self, 0.0, L_beam)]   # self-weight acts DOWN only

R_vert,  V_vert_arr,  M_vert_arr,  y_vert  = solve_beam(vert_pts,  vert_dist,  I_x)  # bends about x (strong)
R_horiz, V_horiz_arr, M_horiz_arr, y_horiz = solve_beam(horiz_pts, horiz_dist, I_y)  # bends about y (weak)

V_vert,  _        = peak(V_vert_arr);  M_vert,  xM_vert  = peak(M_vert_arr)
V_horiz, _        = peak(V_horiz_arr); M_horiz, xM_horiz = peak(M_horiz_arr)

print(f"CASE = {CASE},  theta = {np.degrees(theta):.2f} deg")
print(f"F_y = {F_y:.0f} lb,  F_x = {F_x:.0f} lb,  N = {F:.0f} lb,  friction = {Friction_force:.0f} lb\n")
print(f"{'':12}{'R1':>9}{'R2':>9}{'R3':>9}{'R4':>9}{'|V|max':>10}{'|M|max':>12}")
for name, Rs, Vp, Mp, xM in [("Vertical", R_vert, V_vert, M_vert, xM_vert),
                             ("Horizontal", R_horiz, V_horiz, M_horiz, xM_horiz)]:
    print(f"{name:12}" + "".join(f"{r_:9.1f}" for r_ in Rs) + f"{Vp:10.0f}{Mp:12.0f}   (M at x = {xM:.1f} in)")

if PLOT:
    fig, axs = plt.subplots(2, 2, sharex=True, figsize=(12, 7))
    for col, (V, M, name, c) in enumerate([(V_vert_arr, M_vert_arr, "Vertical (strong axis)", "tab:blue"),
                                           (V_horiz_arr, M_horiz_arr, "Horizontal (weak axis)", "tab:orange")]):
        for row, arr in enumerate((V, M)):
            ax = axs[row, col]
            ax.fill_between(xs, arr, alpha=.3, color=c); ax.plot(xs, arr, color=c)
            ax.axhline(0, color="k", lw=.8)
            for s in ties: ax.axvline(s, color="gray", ls=":")
        axs[0, col].set_title(name); axs[1, col].set_xlabel("x along beam (in)")
    axs[0, 0].set_ylabel("V (lb)"); axs[1, 0].set_ylabel("M (lb*in)")
    fig.suptitle(f"{CASE} case: vertical R = {', '.join(f'{r_:.0f}' for r_ in R_vert)} lb;  "
                 f"horizontal R = {', '.join(f'{r_:.0f}' for r_ in R_horiz)} lb")
    plt.tight_layout(); plt.show()

#%%
#-------------------------Stress Calculations--------------------------
stress_axial       = Friction_force / A                    # friction along the beam
stress_bending_x   = M_vert / S_x                          # vertical load -> strong axis
stress_bending_y   = M_horiz / S_y                         # sideways load -> weak axis
stress_bending_max = stress_bending_x + stress_bending_y   # both peak at the same corner
stress_shear_vert  = V_vert  * Q_x / (I_x * 2*t)           # through the two side walls
stress_shear_horiz = V_horiz * Q_y / (I_y * 2*t)           # through top/bottom walls

if CASE == "point":
    T = F_x*(h/2) - F_y*(b/2)                              # lb*in, whole load at one corner
else:
    T = q1*((b/2) - np.tan(theta)*(h/2)) * (L1/2)          # lb*in, spread load over half of section 1
stress_torsion = T / (2*A_m*t)                             # Bredt, thin closed tube

def show(label, stress, allow):
    print(f"{label:31}= {stress:6.0f} psi  |  FS = {allow/stress:.2f}")

show("Axial (friction)",              stress_axial,       Fy)
show("Bending, strong axis (x-axis)", stress_bending_x,   Fy)
show("Bending, weak axis (y-axis)",   stress_bending_y,   Fy)
show("Bending, maximum",              stress_bending_max, Fy)
show("Shear, vertical",               stress_shear_vert,  Fy_shear)
show("Shear, horizontal",             stress_shear_horiz, Fy_shear)
print(f"{'Torque T':31}= {T:6.0f} lb*in")
show("Torsional shear",               stress_torsion,     Fy_shear)

#-------------------------Deflection (Macaulay curve, beam on 4 ties)-------------------------
d_vert,  xd_vert  = peak(y_vert)
d_horiz, xd_horiz = peak(y_horiz)
print(f"\nDeflection, vertical   = {d_vert:.4f} in at x = {xd_vert:.1f} in  |  limit L/360 = {span_max/360:.3f} in")
print(f"Deflection, horizontal = {d_horiz:.4f} in at x = {xd_horiz:.1f} in")

#-------------------------Buckling-------------------------
# 1. Column buckling (AISC E3): friction push, weak axis, longest tie span, K = 1
slender = span_max / r_y
Fe  = np.pi**2 * E / slender**2
Fcr = 0.658**(Fy/Fe)*Fy if slender <= 4.71*np.sqrt(E/Fy) else 0.877*Fe
P_n = Fcr * A
print(f"\nColumn buckling:   KL/r = {slender:.1f}, Fe = {Fe:.0f} psi, Fcr = {Fcr:.0f} psi, "
      f"Pn = {P_n:.0f} lb  |  FS = {P_n/Friction_force:.1f}")

# 2. Local wall buckling (B4.1) and 3. shear buckling of the webs (G4, kv = 5)
lam_flange, lam_web, lam_shear = 1.12*np.sqrt(E/Fy), 2.42*np.sqrt(E/Fy), 1.10*np.sqrt(5*E/Fy)
print(f"Local buckling:    b/t = {b_t:.1f} < {lam_flange:.1f}: {b_t < lam_flange},  "
      f"h/t = {h_t:.1f} < {lam_web:.1f}: {h_t < lam_web}")
print(f"Shear buckling:    h/t = {h_t:.1f} < {lam_shear:.1f}: {h_t < lam_shear}")

# 4. Web local yielding under the concentrated force (J10.2), two webs
l_b = 2.0                                                  # in, assumed bearing length
R_n_web = 2 * Fy * t * (5*1.5*t + l_b)                     # k = 1.5t
P_web = F_y if CASE == "point" else max(R_vert)
print(f"Web yielding:      Rn = {R_n_web:.0f} lb vs {P_web:.0f} lb  |  FS = {R_n_web/P_web:.1f}")

# 5. Lateral-torsional buckling (F7): does not apply if Lp > span
L_p = 0.13 * E * r_y * np.sqrt(J*A) / (Fy*Z_x)
print(f"Lateral-torsional: Lp = {L_p:.0f} in > span {span_max:.1f} in: {L_p > span_max} (does not apply)")

#-------------------------Von Mises & Tresca-------------------------
sigma = stress_axial + stress_bending_max                  # same sigma at both points (conservative)
header = f"\n{'-'*25}Von-Mises & Tresca{'-'*25}"
print(header)
for name, tau in {"Web (side walls)":    stress_shear_vert  + stress_torsion,
                  "Flange (top/bottom)": stress_shear_horiz + stress_torsion}.items():
    radius = np.hypot(sigma/2, tau)
    sigma_1, sigma_2 = sigma/2 + radius, sigma/2 - radius
    tresca    = max(abs(sigma_1 - sigma_2), abs(sigma_1), abs(sigma_2))
    von_mises = np.sqrt(sigma_1**2 - sigma_1*sigma_2 + sigma_2**2)
    #print(tresca, von_mises)
    print(f"{name}: tau = {tau:.0f} psi, sigma_1 = {sigma_1:.0f}, sigma_2 = {sigma_2:.0f} psi")
    print(f"   Tresca    = {tresca:.0f} psi  |  FS = {Fy/tresca:.2f}")
    print(f"   Von Mises = {von_mises:.0f} psi  |  FS = {Fy/von_mises:.2f}")
    print("-" * (len(header) - 1))
# %%
#-------------------------Tipping + Slipping Calculations-------------------------

#-------------------------Human Variables-------------------------
human_height = 70 #in           #average hight of a man (5'10")
human_weight = 200              #lb, average weight of a college man (180 lb)
lean_angle = np.radians(20)     #deg, upper limit of the lean angle majority of people make (5<x<20)
lean_force_static = ((.56*human_height)/(.82*human_height)) * human_weight * np.tan(lean_angle)  #lb, force required to push a man over at a given lean angle
d_stop = 8 #in                  #stopping distance of a human hitting a wall
lean_force_kinetic = ((human_weight * human_height)*(1-np.cos(lean_angle))) / d_stop
avg_push_force = 60 #lb         #This push force accounts for the average person attempting to push without bracing of any sorts
max_push_force = 160 #lb        #Max force of a person attempting to push the machine
push_force = [avg_push_force, max_push_force]
lean_force = [lean_force_static, lean_force_kinetic]
#-------------------------Cradle only: Pushing and slipping-------------------------
F_push_top_cradle = (w_tbm_lb*(d_opening/2))/d_in  #lb        #worst case: pushed from the side at the top of the cradle
F_push_mid_cradle = (w_tbm_lb*(d_opening/2))/(d_in/2)  #lb    #worst case: pushed from the side at the mid-height of the cradle
print(f"{'-'*25}Cradle pushing{'-'*25}")
type_push = ['without', 'with']
type_trying = ['(not trying)', '(trying)']
x = 0
for push in push_force:
    print(f"Force required to push the TBM out of the cradle with a force of {push}lb from:")
    print(f"   The top-most part of the TBM:  {F_push_top_cradle:.1f} lb | FS = {F_push_top_cradle/push:.2f}")
    print(f"   The mid-height of the TBM:     {F_push_mid_cradle:.1f} lb | FS = {F_push_mid_cradle/push:.2f}")
    print(f"This run uses the average push force of a human", type_push[x], "bracing", type_trying[x])
    print(f"{'-'*75}")
    x += 1
    #print("Scenario 1 assumes that the person pushing as a perfect grip at the top of the TBM (which should never be the case due to sliding/friction)")
    #print("Scenario 2 assumes that the person pushes perpendicularly at the mid-point")

Frictional_torque = Friction_force*r
Inertia = .25*w_tbm_lb*r**2
print(f"\n{'-'*25}Tangential force required to spin the TBM{'-'*25}")
print(f"force required to spin the TBM:", Frictional_torque, 'lb')

#------------------------Cardle + Table: tipping and slipping--------------------
w_cradle_kg = 107 #kg                 #weight of cradle
w_cradle_lb = w_cradle_kg*2.20462 #lb #weight of cradle (235.9 lb)
w_table_kg = 252.5 #kg                #weight of table
w_table_lb = w_table_kg*2.20462 #lb   #weight of table (556.7 lb)
width_table_legs = 24.25 #in          #wight from inside to inside of wheels (28.4in outside to outside)
height_table = 33.25 #in              #height of table            
F_tip_top_table = ((w_tbm_lb + w_cradle_lb + w_table_lb)*(width_table_legs/2)) / (height_table + 4 + d_in)
F_tip_mid_table = ((w_tbm_lb + w_cradle_lb + w_table_lb)*(width_table_legs/2)) / (height_table + 4 + r)

print(f"\n{'-'*25}Cardle + Table: tipping and slipping{'-'*25}")
type_lean = ('static', 'kinetic')
x = 0
for lean in lean_force:
    print(f"Force required to tip the table from", type_lean[x], "leaning from:")
    print(f"   The top-most part of the TBM ({(height_table + 4 + d_in):.1f}in):  {F_tip_top_table:.1f} lb | FS = {(F_tip_top_table/lean):.2f}")
    print(f"   The mid-height of the TBM ({(height_table + 4 + r):.1f}in):     {F_tip_mid_table:.1f} lb | FS = {(F_tip_mid_table/lean):.2f}")
    print(f"{'-'*75}")
    x += 1


# %%
