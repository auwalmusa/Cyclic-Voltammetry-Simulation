import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io

# Define the CV simulation function (Standard Model)
def run_cv_simulation(C, D, etai, etaf, v, n, alpha, k0, kc, T):
    """
    Cyclic voltammetry simulation for EC mechanism
    Based on Bard and Faulkner, Appendix B
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
        for i2 in range(1, j-1):
            O[i1+1, i2] = O[i1, i2] + DM * (O[i1, i2+1] + O[i1, i2-1] - 2*O[i1, i2])
            R[i1+1, i2] = R[i1, i2] + DM * (R[i1, i2+1] + R[i1, i2-1] - 2*R[i1, i2]) - km * R[i1, i2]
        
        JO[i1+1] = (kf[i1+1] * O[i1+1, 1] - kb[i1+1] * R[i1+1, 1]) / (1 + Dx/D * (kf[i1+1] + kb[i1+1]))
        O[i1+1, 0] = O[i1+1, 1] - JO[i1+1] * (Dx/D)
        R[i1+1, 0] = R[i1+1, 1] + JO[i1+1] * (Dx/D) - km * R[i1+1, 0]
    
    Z = -n * F * JO * 1000  # [A/cm^2 -> mA/cm^2]
    if len(eta) > len(Z):
        eta = eta[:len(Z)]
    
    return eta, Z

# Helper function to convert data to CSV
def convert_to_csv(data, header='Potential (V),Current'):
    csv_io = io.StringIO()
    np.savetxt(csv_io, data, delimiter=',', header=header, comments='')
    return csv_io.getvalue()

# Main Streamlit app
def create_streamlit_app():
    st.set_page_config(page_title="Cyclic Voltammetry Simulator", layout="wide")
    st.title("Cyclic Voltammetry Simulator")
    st.write("Based on Bard and Faulkner, Appendix B - EC mechanism")
    
    # Define top-level tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Simulator", "Parameter Study", "Compare Models", "Educational"])
    
    # Tab 1: Simulator
    with tab1:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Simulation Parameters")
            
            C = st.number_input("Concentration [mol/L]", min_value=0.01, max_value=10.0, value=1.0, step=0.1)
            D = st.number_input("Diffusion coef. [cm²/s]", min_value=1e-6, max_value=1e-4, value=1e-5, format="%.1e", step=1e-6)
            
            st.subheader("Potential Settings")
            etai = st.slider("Initial E [V]", min_value=-1.0, max_value=1.0, value=0.2, step=0.1)
            etaf = st.slider("Final E [V]", min_value=-1.0, max_value=1.0, value=-0.2, step=0.1)
            v = st.number_input("Sweep rate [V/s]", min_value=1e-4, max_value=1.0, value=1e-3, format="%.1e", step=1e-4)
            
            with st.expander("Advanced Parameters"):
                n = st.number_input("Electrons transferred", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
                alpha = st.number_input("Transfer coefficient", min_value=0.1, max_value=0.9, value=0.5, step=0.1)
                k0 = st.number_input("Rate constant [cm/s]", min_value=1e-5, max_value=1.0, value=1e-2, format="%.1e", step=1e-3)
                kc = st.number_input("Chemical rate [1/s]", min_value=1e-5, max_value=10.0, value=1e-3, format="%.1e", step=1e-4)
                T = st.number_input("Temperature [K]", min_value=273.15, max_value=373.15, value=298.15, step=5.0)
        
        with col2:
            eta, current = run_cv_simulation(C, D, etai, etaf, v, n, alpha, k0, kc, T)
            y_label = "Current density (mA/cm²)"
            title = "Cyclic Voltammogram (Standard Model)"
            file_name = "cv_simulation_data.csv"
            header = 'Potential (V),Current (mA/cm²)'
            
            # Display key parameters
            tk = 2*(etai-etaf)/v
            ktk = kc*tk
            km = ktk/500  # L=500
            F = 96485
            R = 8.3145
            f = F/(R*T)
            Lambda = k0/(D*f*v)**0.5
            
            param_col1, param_col2, param_col3 = st.columns(3)
            with param_col1:
                st.metric("Kinetic param (k·tₖ)", f"{ktk:.2e}")
            with param_col2:
                st.metric("Norm. kinetic (k·tₖ/L)", f"{km:.2e}", delta="Warning!" if km > 0.1 else "OK", delta_color="off" if km <= 0.1 else "red")
            with param_col3:
                st.metric("Reversibility (Λ)", f"{Lambda:.2e}")
            
            # Plot the results
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(eta, current)
            ax.set_xlabel('Overpotential (V)')
            ax.set_ylabel(y_label)
            ax.set_title(title)
            ax.grid(True, linestyle='--', alpha=0.7)
            fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
            st.pyplot(fig)
            
            # Download button
            csv_data = np.column_stack((eta, current))
            st.download_button(
                label="Download Data as CSV",
                data=convert_to_csv(csv_data, header),
                file_name=file_name,
                mime="text/csv"
            )
    
    # Tab 2: Parameter Study (placeholder)
    with tab2:
        st.write("Parameter Study functionality to be implemented here.")
    
    # Tab 3: Compare Models (placeholder)
    with tab3:
        st.write("Compare Models functionality to be implemented here.")
    
    # Tab 4: Educational Resources
    with tab4:
        st.subheader("Educational Resources")
        
        edu_tab1, edu_tab2, edu_tab3 = st.tabs(["CV Basics", "EC Mechanism", "Interactive Demos"])
        
        with edu_tab1:
            st.markdown("""
            ## Cyclic Voltammetry Fundamentals
            
            Cyclic voltammetry (CV) is an electrochemical technique where the working electrode potential is ramped linearly versus time. When the potential reaches a set value, the scan is reversed to return to the initial potential.
            
            ### Key Components:
            
            **The CV waveform:** 
            """)
            
            fig, ax = plt.subplots(figsize=(6, 3))
            time = np.linspace(0, 10, 1000)
            potential = np.concatenate([np.linspace(0, 0.5, 250), np.linspace(0.5, -0.5, 500), np.linspace(-0.5, 0, 250)])
            ax.plot(time, potential)
            ax.set_xlabel('Time')
            ax.set_ylabel('Potential (V)')
            ax.set_title('Cyclic Voltammetry Potential Waveform')
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)
            
            st.markdown("""
            ### The CV Response:
            
            The resulting current-potential plot is called a cyclic voltammogram, which is characterized by:
            
            1. **Anodic peak:** Corresponds to oxidation (loss of electrons)
            2. **Cathodic peak:** Corresponds to reduction (gain of electrons)
            
            ### Diagnostic Parameters:
            
            * **Peak potential separation (ΔEp):** For a reversible system, ΔEp = 59 mV/n (at 25°C)
            * **Peak current ratio (ipc/ipa):** For a reversible system, ipc/ipa ≈ 1
            * **Peak current:** Proportional to concentration and square root of scan rate
            
            ### The Randles-Sevcik Equation:
            
            For a reversible system at 25°C:
            
            $i_p = 0.4463 \\cdot n F A C \\sqrt{\\frac{nFvD}{RT}}$
            
            Where:
            - ip = peak current (A)
            - n = number of electrons transferred
            - F = Faraday constant (96,485 C/mol)
            - A = electrode area (cm²)
            - C = concentration (mol/cm³)
            - v = scan rate (V/s)
            - D = diffusion coefficient (cm²/s)
            - R = gas constant (8.314 J/mol·K)
            - T = temperature (K)
            """)
        
        with edu_tab2:
            st.markdown("""
            ## EC Mechanism in Cyclic Voltammetry
            
            The EC mechanism consists of an Electrochemical reaction followed by a Chemical reaction:
            
            **Step 1 (E):** O + e⁻ ⟷ R (electrochemical, reversible)  
            **Step 2 (C):** R → Z (chemical, irreversible)
            
            ### Impact on Cyclic Voltammogram:
            
            The following chemical reaction consumes the electrochemically generated species R, which:
            
            1. Reduces the reverse peak current
            2. Shifts the peak potentials
            3. Changes the peak current ratio (ipr/ipf < 1)
            
            ### Characteristic Features:
            """)
            
            fig, ax = plt.subplots(figsize=(7, 4))
            potential = np.linspace(0.5, -0.5, 1000)
            current_rev = -10 * (np.exp(-40*(potential-0.05)) - np.exp(-40*(potential+0.05))) / (1 + np.exp(-40*(potential-0.05)) + np.exp(-40*(potential+0.05)))
            current_ec = -10 * (np.exp(-40*(potential-0.05)) - 0.6*np.exp(-40*(potential+0.05))) / (1 + np.exp(-40*(potential-0.05)) + np.exp(-40*(potential+0.05)))
            ax.plot(potential, current_rev, 'b-', label='Reversible')
            ax.plot(potential, current_ec, 'r--', label='EC Mechanism')
            ax.set_xlabel('Potential (V)')
            ax.set_ylabel('Current')
            ax.set_title('Comparison: Reversible vs. EC Mechanism')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            st.pyplot(fig)
            
            st.markdown("""
            ### Diagnostic Criteria:
            
            1. **Rate of following reaction (kc):**
               - A dimensionless parameter k·tₖ characterizes the influence of the chemical reaction
               - k·tₖ > 10: Strong chemical reaction effect (reverse peak greatly diminished)
               - 0.1 < k·tₖ < 10: Moderate effect (reverse peak partially diminished)
               - k·tₖ < 0.1: Weak effect (system approaches reversible behavior)
            
            2. **Peak current ratio (ipr/ipf):**
               - Decreases as k·tₖ increases
               - Used to calculate the rate constant of the chemical reaction
            
            3. **Scan rate dependency:**
               - At faster scan rates, there's less time for the chemical reaction to occur
               - EC behavior can appear more reversible at high scan rates
            """)
        
        with edu_tab3:
            st.markdown("## Interactive Demonstrations")
            
            demo_type = st.selectbox(
                "Select a demonstration:",
                ["Effect of scan rate", "Effect of chemical reaction rate", "Effect of electron transfer rate"]
            )
            
            if demo_type == "Effect of scan rate":
                st.markdown("""
                ### Effect of Scan Rate on EC Mechanism
                
                Adjust the slider to see how scan rate affects the cyclic voltammogram for a system with an EC mechanism:
                """)
                
                sweep_rate = st.slider("Scan rate (V/s)", min_value=0.001, max_value=1.0, value=0.01, step=0.01, format="%.3f")
                demo_C, demo_D, demo_etai, demo_etaf, demo_n, demo_alpha, demo_k0, demo_kc, demo_T = 1.0, 1e-5, 0.5, -0.5, 1.0, 0.5, 1e-2, 0.5, 298.15
                eta_demo, current_demo = run_cv_simulation(demo_C, demo_D, demo_etai, demo_etaf, sweep_rate, demo_n, demo_alpha, demo_k0, demo_kc, demo_T)
                tk_demo = 2*(demo_etai-demo_etaf)/sweep_rate
                ktk_demo = demo_kc*tk_demo
                
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(eta_demo, current_demo)
                ax.set_xlabel('Potential (V)')
                ax.set_ylabel('Current density (mA/cm²)')
                ax.set_title(f'EC Mechanism (Scan rate = {sweep_rate} V/s, k·tₖ = {ktk_demo:.2f})')
                ax.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig)
                
                st.markdown(f"""
                **Observations:**
                
                - At this scan rate ({sweep_rate} V/s), the dimensionless parameter k·tₖ = {ktk_demo:.2f}
                - {'Higher scan rates decrease k·tₖ, making the system appear more reversible' if sweep_rate > 0.1 else 'Lower scan rates increase k·tₖ, enhancing the effect of the chemical reaction'}
                - {'At very high scan rates, there is less time for the chemical reaction to occur, so the reverse peak becomes more prominent' if sweep_rate > 0.5 else ''}
                """)
            
            elif demo_type == "Effect of chemical reaction rate":
                st.markdown("""
                ### Effect of Chemical Reaction Rate on EC Mechanism
                
                Adjust the slider to see how the chemical reaction rate constant (kc) affects the cyclic voltammogram:
                """)
                
                chem_rate = st.slider("Chemical reaction rate constant (s⁻¹)", min_value=0.001, max_value=10.0, value=0.5, step=0.1, format="%.3f")
                demo_C, demo_D, demo_etai, demo_etaf, demo_sweep_rate, demo_n, demo_alpha, demo_k0, demo_T = 1.0, 1e-5, 0.5, -0.5, 0.05, 1.0, 0.5, 1e-2, 298.15
                eta_demo, current_demo = run_cv_simulation(demo_C, demo_D, demo_etai, demo_etaf, demo_sweep_rate, demo_n, demo_alpha, demo_k0, chem_rate, demo_T)
                tk_demo = 2*(demo_etai-demo_etaf)/demo_sweep_rate
                ktk_demo = chem_rate*tk_demo
                
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(eta_demo, current_demo)
                ax.set_xlabel('Potential (V)')
                ax.set_ylabel('Current density (mA/cm²)')
                ax.set_title(f'EC Mechanism (kc = {chem_rate} s⁻¹, k·tₖ = {ktk_demo:.2f})')
                ax.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig)
                
                st.markdown(f"""
                **Observations:**
                
                - At this chemical rate constant ({chem_rate} s⁻¹), the dimensionless parameter k·tₖ = {ktk_demo:.2f}
                - {'Higher kc values lead to stronger EC effects, diminishing the reverse peak' if chem_rate > 1.0 else 'Lower kc values make the system approach reversible behavior'}
                - {'At very high kc values, species R is rapidly consumed, leading to complete disappearance of the reverse peak' if chem_rate > 5.0 else ''}
                """)
            
            else:  # Effect of electron transfer rate
                st.markdown("""
                ### Effect of Electron Transfer Rate on CV Response
                
                Adjust the slider to see how the electron transfer rate constant (k₀) affects the cyclic voltammogram:
                """)
                
                et_rate = st.slider("Electron transfer rate constant (cm/s)", min_value=1e-5, max_value=1e-1, value=1e-3, format="%.1e")
                demo_C, demo_D, demo_etai, demo_etaf, demo_sweep_rate, demo_n, demo_alpha, demo_kc, demo_T = 1.0, 1e-5, 0.5, -0.5, 0.05, 1.0, 0.5, 0.01, 298.15
                eta_demo, current_demo = run_cv_simulation(demo_C, demo_D, demo_etai, demo_etaf, demo_sweep_rate, demo_n, demo_alpha, et_rate, demo_kc, demo_T)
                F, R, f = 96485, 8.3145, F/(R*demo_T)
                Lambda = et_rate/(demo_D*f*demo_sweep_rate)**0.5
                
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(eta_demo, current_demo)
                ax.set_xlabel('Potential (V)')
                ax.set_ylabel('Current density (mA/cm²)')
                ax.set_title(f'Effect of k₀ = {et_rate:.1e} cm/s (Λ = {Lambda:.2f})')
                ax.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig)
                
                reversibility = "Reversible" if Lambda > 15 else "Quasi-reversible" if Lambda > 1e-3 else "Irreversible"
                st.markdown(f"""
                **Observations:**
                
                - At this electron transfer rate ({et_rate:.1e} cm/s), the dimensionless parameter Λ = {Lambda:.2f}
                - The system is classified as: **{reversibility}**
                - {'Higher k₀ values lead to more reversible behavior with smaller peak separation' if et_rate > 1e-2 else 'Lower k₀ values lead to irreversible behavior with larger peak separation'}
                - {'The peaks are nearly symmetric and the peak separation approaches 59 mV/n' if Lambda > 15 else ''}
                """)

if __name__ == "__main__":
    create_streamlit_app()
