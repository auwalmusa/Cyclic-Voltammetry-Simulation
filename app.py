import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def run_cv_simulation(C, D, etai, etaf, v, n, alpha, k0, kc, T):
    """
    Cyclic voltammetry simulation for EC mechanism
    Based on Bard and Faulkner, Appendix B
    
    Parameters:
    C      : float, initial concentration of O [mol/L]
    D      : float, diffusion coefficient [cm^2/s]
    etai   : float, initial overpotential [V]
    etaf   : float, final overpotential [V]
    v      : float, sweep rate [V/s]
    n      : float, number of electrons transferred
    alpha  : float, dimensionless charge-transfer coefficient
    k0     : float, electrochemical rate constant [cm/s]
    kc     : float, chemical rate constant [1/s]
    T      : float, temperature [K]
    
    Returns:
    eta    : array, overpotential values [V]
    Z      : array, current density [mA/cm^2]
    """
    # Physical constants
    F = 96485    # [C/mol], Faraday's constant
    R = 8.3145   # [J/mol-K], ideal gas constant
    f = F/(R*T)  # [1/V], normalized Faraday's constant at room temperature
    
    # Simulation variables
    L = 500      # number of iterations per t_k
    DM = 0.45    # model diffusion coefficient
    
    # Derived constants
    tk = 2*(etai-etaf)/v    # [s], characteristic exp. time
    Dt = tk/L               # [s], delta time
    Dx = np.sqrt(D*Dt/DM)   # [cm], delta x
    j = int(np.ceil(4.2*L**0.5)+5)  # number of boxes
    
    # Reversibility parameters
    ktk = kc*tk             # dimensionless kinetic parameter
    km = ktk/L              # normalized dimensionless kinetic parameter
    Lambda = k0/(D*f*v)**0.5  # dimensionless reversibility parameter
    
    # Chemical reversibility warning
    if km > 0.1:
        st.warning(f"k_c*t_k/l equals {km:.3f}, which exceeds the upper limit of 0.1 (see B&F, pg 797)")
    
    # Convert C from mol/L to mol/cm3
    C = C / 1000
    
    # Time and potential arrays
    k_array = np.arange(L+1)
    t = Dt * k_array
    eta1 = etai - v*t
    eta2 = etaf + v*t
    
    # Combine forward and reverse scans
    eta_fwd = eta1[eta1 > etaf]
    eta_rev = eta2[eta2 <= etai]
    eta = np.concatenate((eta_fwd, eta_rev))
    
    # Normalized overpotential and rate constants
    Enorm = eta * f
    kf = k0 * np.exp(-alpha * n * Enorm)
    kb = k0 * np.exp((1-alpha) * n * Enorm)
    
    # Initialize concentration arrays
    O = C * np.ones((L+1, j))
    R = np.zeros((L+1, j))
    JO = np.zeros(L+1)
    
    # Main simulation loop
    for i1 in range(L):
        # Update bulk concentrations of O and R
        for i2 in range(1, j-1):
            O[i1+1, i2] = O[i1, i2] + DM * (O[i1, i2+1] + O[i1, i2-1] - 2*O[i1, i2])
            R[i1+1, i2] = R[i1, i2] + DM * (R[i1, i2+1] + R[i1, i2-1] - 2*R[i1, i2]) - km * R[i1, i2]
        
        # Update flux
        JO[i1+1] = (kf[i1+1] * O[i1+1, 1] - kb[i1+1] * R[i1+1, 1]) / (1 + Dx/D * (kf[i1+1] + kb[i1+1]))
        
        # Update surface concentrations
        O[i1+1, 0] = O[i1+1, 1] - JO[i1+1] * (Dx/D)
        R[i1+1, 0] = R[i1+1, 1] + JO[i1+1] * (Dx/D) - km * R[i1+1, 0]
    
    # Calculate current density from flux of O
    Z = -n * F * JO * 1000  # [A/cm^2 -> mA/cm^2]
    
    # Ensure eta and Z are the same length
    if len(eta) > len(Z):
        eta = eta[:len(Z)]
    
    return eta, Z

def run_dimensionless_cv_simulation(C, D, etai, etaf, v, n, alpha, k0, kc, T):
    """
    Dimensionless cyclic voltammetry simulation for EC mechanism
    Based on Bard and Faulkner, Appendix B - dimensionless formulation
    
    Parameters:
    C      : float, initial concentration of O [mol/L]
    D      : float, diffusion coefficient [cm^2/s]
    etai   : float, initial overpotential [V]
    etaf   : float, final overpotential [V]
    v      : float, sweep rate [V/s]
    n      : float, number of electrons transferred
    alpha  : float, dimensionless charge-transfer coefficient
    k0     : float, electrochemical rate constant [cm/s]
    kc     : float, chemical rate constant [1/s]
    T      : float, temperature [K]
    
    Returns:
    eta    : array, overpotential values [V]
    current: array, dimensionless current
    """
    # Physical constants
    F = 96485    # [C/mol], Faraday's constant
    R = 8.3145   # [J/mol-K], ideal gas constant
    f = F/(R*T)  # [1/V], normalized Faraday's constant at room temperature
    
    # Simulation variables
    L = 500      # number of iterations per t_k
    DM = 0.45    # model diffusion coefficient
    
    # Derived constants
    tk = 2*(etai-etaf)/v    # [s], characteristic exp. time
    Dt = tk/L               # [s], delta time
    Dx = np.sqrt(D*Dt/DM)   # [cm], delta x
    j = int(np.ceil(4.2*L**0.5)+5)  # number of boxes
    
    # Reversibility parameters
    ktk = kc*tk             # dimensionless kinetic parameter
    km = ktk/L              # normalized dimensionless kinetic parameter
    Lambda = k0/(D*f*v)**0.5  # dimensionless reversibility parameter
    
    # Chemical reversibility warning
    if km > 0.1:
        st.warning(f"k_c*t_k/l equals {km:.3f}, which exceeds the upper limit of 0.1 (see B&F, pg 797)")
    
    # Convert C from mol/L to mol/cm3
    C = C / 1000
    
    # Time and potential arrays
    k_array = np.arange(L+1)
    t = Dt * k_array
    eta1 = etai - v*t
    eta2 = etaf + v*t
    
    # Combine forward and reverse scans
    eta_fwd = eta1[eta1 > etaf]
    eta_rev = eta2[eta2 <= etai]
    eta = np.concatenate((eta_fwd, eta_rev))
    
    # Normalized overpotential and dimensionless rate constants
    Enorm = eta * f
    kf = (k0*(tk/D)**0.5) * np.exp(-alpha * n * Enorm)
    kb = (k0*(tk/D)**0.5) * np.exp((1-alpha) * n * Enorm)
    
    # Initialize concentration arrays
    O = C * np.ones((L+1, j))
    R = np.zeros((L+1, j))
    Z = np.zeros(L+1)  # dimensionless flux of O at the surface
    
    # Main simulation loop
    for i1 in range(L+1):
        # Update flux
        Z[i1] = (kf[i1] * O[i1, 1] - kb[i1] * R[i1, 1]) / (1 + kf[i1] + kb[i1])
        
        # Update surface concentrations
        O[i1, 0] = O[i1, 1] - Z[i1]
        R[i1, 0] = R[i1, 1] + Z[i1] - km * R[i1, 0]
        
        # Update bulk concentrations of O and R (for next time step)
        if i1 < L:  # Only update if not the last iteration
            for i2 in range(1, j-1):
                O[i1+1, i2] = O[i1, i2] + DM * (O[i1, i2+1] + O[i1, i2-1] - 2*O[i1, i2])
                R[i1+1, i2] = R[i1, i2] + DM * (R[i1, i2+1] + R[i1, i2-1] - 2*R[i1, i2]) - km * R[i1, i2]
    
    # Ensure eta and Z are the same length
    if len(eta) > len(Z):
        eta = eta[:len(Z)]
    
    # Convert dimensionless current to physical current (multiplied by 16 as in MATLAB code)
    current = -Z * 16
    
    return eta, current

def create_streamlit_app():
    st.set_page_config(page_title="Cyclic Voltammetry Simulator", layout="centered")
    
    st.title("Cyclic Voltammetry Simulation")
    st.write("Based on Bard and Faulkner, Appendix B - EC mechanism")
    
    # Model selection
    model_type = st.radio(
        "Select simulation model:",
        ["Standard (Current Density)", "Dimensionless"],
        horizontal=True
    )
    
    st.sidebar.header("Simulation Parameters")
    
    # Independent variables with sliders and input boxes
    C = st.sidebar.number_input("Initial concentration of O [mol/L]", min_value=0.01, max_value=10.0, value=1.0, step=0.1)
    D = st.sidebar.number_input("Diffusion coefficient [cm²/s]", min_value=1e-6, max_value=1e-4, value=1e-5, format="%.1e", step=1e-6)
    etai = st.sidebar.slider("Initial overpotential [V]", min_value=-1.0, max_value=1.0, value=0.2, step=0.1)
    etaf = st.sidebar.slider("Final overpotential [V]", min_value=-1.0, max_value=1.0, value=-0.2, step=0.1)
    v = st.sidebar.number_input("Sweep rate [V/s]", min_value=1e-4, max_value=1.0, value=1e-3, format="%.1e", step=1e-4)
    
    # Advanced parameters (collapsible)
    with st.sidebar.expander("Advanced Parameters"):
        n = st.number_input("Number of electrons transferred", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        alpha = st.number_input("Charge-transfer coefficient", min_value=0.1, max_value=0.9, value=0.5, step=0.1)
        k0 = st.number_input("Electrochemical rate constant [cm/s]", min_value=1e-5, max_value=1.0, value=1e-2, format="%.1e", step=1e-3)
        kc = st.number_input("Chemical rate constant [1/s]", min_value=1e-5, max_value=1.0, value=1e-3, format="%.1e", step=1e-4)
        T = st.number_input("Temperature [K]", min_value=273.15, max_value=373.15, value=298.15, step=5.0)
    
    # Run simulation based on selected model
    if model_type == "Standard (Current Density)":
        eta, current = run_cv_simulation(C, D, etai, etaf, v, n, alpha, k0, kc, T)
        y_label = "Current density (mA/cm²)"
        title = "Cyclic Voltammogram (Standard Model)"
        file_name = "cv_simulation_data.csv"
        header = 'Potential (V),Current (mA/cm²)'
    else:
        eta, current = run_dimensionless_cv_simulation(C, D, etai, etaf, v, n, alpha, k0, kc, T)
        y_label = "Dimensionless current"
        title = "Cyclic Voltammogram (Dimensionless Formulation)"
        file_name = "dimensionless_cv_simulation_data.csv"
        header = 'Potential (V),Dimensionless Current'
    
    # Display reversibility parameters
    tk = 2*(etai-etaf)/v
    ktk = kc*tk
    km = ktk/500  # L=500
    F = 96485
    R = 8.3145
    f = F/(R*T)
    Lambda = k0/(D*f*v)**0.5
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Kinetic parameter (k·tₖ)", f"{ktk:.3e}")
    with col2:
        st.metric("Normalized kinetic parameter (k·tₖ/L)", f"{km:.3e}")
    with col3:
        st.metric("Reversibility parameter (Λ)", f"{Lambda:.3e}")
    
    # Plot the results
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(eta, current)
    ax.set_xlabel('Overpotential (V)')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add maximum and minimum current indicators
    max_current_idx = np.argmax(current)
    min_current_idx = np.argmin(current)
    
    ax.plot(eta[max_current_idx], current[max_current_idx], 'ro')
    ax.annotate(f'({eta[max_current_idx]:.2f}V, {current[max_current_idx]:.2f})', 
                xy=(eta[max_current_idx], current[max_current_idx]),
                xytext=(eta[max_current_idx], current[max_current_idx] + (max(current) - min(current))*0.05),
                ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    
    ax.plot(eta[min_current_idx], current[min_current_idx], 'ro')
    ax.annotate(f'({eta[min_current_idx]:.2f}V, {current[min_current_idx]:.2f})', 
                xy=(eta[min_current_idx], current[min_current_idx]),
                xytext=(eta[min_current_idx], current[min_current_idx] - (max(current) - min(current))*0.05),
                ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    
    # Plot settings
    plt.tight_layout()
    st.pyplot(fig)
    
    # Download options
    csv_data = np.column_stack((eta, current))
    st.download_button(
        label="Download Data as CSV",
        data=convert_to_csv(csv_data, header),
        file_name=file_name,
        mime="text/csv"
    )
    
    # Add explanation and references
    with st.expander("About this simulation"):
        st.markdown("""
        ### Cyclic Voltammetry Simulation Details
        
        This application simulates cyclic voltammetry experiments for an EC mechanism (electrochemical reaction followed by a chemical reaction) based on the digital simulation approach described in Bard and Faulkner's "Electrochemical Methods: Fundamentals and Applications," Appendix B.
        
        #### Available Models:
        
        1. **Standard Model** - Outputs current density in mA/cm². This uses the standard formulation where the rate constants have physical units.
        
        2. **Dimensionless Model** - Outputs dimensionless current. This model adheres more strictly to the Bard and Faulkner dimensionless variable convention.
        
        #### Key Parameters:
        
        - **Kinetic parameter (k·tₖ)**: Indicates the importance of the following chemical reaction
        - **Normalized kinetic parameter (k·tₖ/L)**: Should be less than 0.1 for accurate simulation
        - **Reversibility parameter (Λ)**: Indicates the electrochemical reversibility
          - Λ > 15: Reversible
          - 15 > Λ > 10⁻³: Quasi-reversible
          - Λ < 10⁻³: Irreversible
        
        #### Reference:
        Bard, A.J.; Faulkner, L.R. Electrochemical Methods: Fundamentals and Applications, 2nd ed.; Wiley: New York, 2001.
        """)

def convert_to_csv(data, header='Potential (V),Current'):
    import io
    csv_io = io.StringIO()
    np.savetxt(csv_io, data, delimiter=',', header=header, comments='')
    return csv_io.getvalue()

if __name__ == "__main__":
    create_streamlit_app()
