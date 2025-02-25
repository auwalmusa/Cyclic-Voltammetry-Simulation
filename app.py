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
            
            # Create example CV waveform
            fig, ax = plt.subplots(figsize=(6, 3))
            time = np.linspace(0, 10, 1000)
            potential = np.concatenate([
                np.linspace(0, 0.5, 250),
                np.linspace(0.5, -0.5, 500),
                np.linspace(-0.5, 0, 250)
            ])
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
            
            # Create comparison of reversible vs EC voltammograms
            fig, ax = plt.subplots(figsize=(7, 4))
            
            # Generate example data
            potential = np.linspace(0.5, -0.5, 1000)
            
            # Simulate reversible response (no chemical reaction)
            current_rev = -10 * (np.exp(-40*(potential-0.05)) - np.exp(-40*(potential+0.05))) / (1 + np.exp(-40*(potential-0.05)) + np.exp(-40*(potential+0.05)))
            
            # Simulate EC mechanism (with chemical reaction)
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
                
                # Create interactive demo
                sweep_rate = st.slider("Scan rate (V/s)", min_value=0.001, max_value=1.0, value=0.01, step=0.01, format="%.3f")
                
                # Fixed parameters
                demo_C = 1.0
                demo_D = 1e-5
                demo_etai = 0.5
                demo_etaf = -0.5
                demo_n = 1.0
                demo_alpha = 0.5
                demo_k0 = 1e-2
                demo_kc = 0.5
                demo_T = 298.15
                
                # Run simulation
                eta_demo, current_demo = run_cv_simulation(demo_C, demo_D, demo_etai, demo_etaf, sweep_rate, demo_n, demo_alpha, demo_k0, demo_kc, demo_T)
                
                # Calculate parameters
                tk_demo = 2*(demo_etai-demo_etaf)/sweep_rate
                ktk_demo = demo_kc*tk_demo
                
                # Create plot
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(eta_demo, current_demo)
                ax.set_xlabel('Potential (V)')
                ax.set_ylabel('Current density (mA/cm²)')
                ax.set_title(f'EC Mechanism (Scan rate = {sweep_rate} V/s, k·tₖ = {ktk_demo:.2f})')
                ax.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig)
                
                # Explanation
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
                
                # Create interactive demo
                chem_rate = st.slider("Chemical reaction rate constant (s⁻¹)", min_value=0.001, max_value=10.0, value=0.5, step=0.1, format="%.3f")
                
                # Fixed parameters
                demo_C = 1.0
                demo_D = 1e-5
                demo_etai = 0.5
                demo_etaf = -0.5
                demo_sweep_rate = 0.05
                demo_n = 1.0
                demo_alpha = 0.5
                demo_k0 = 1e-2
                demo_T = 298.15
                
                # Run simulation
                eta_demo, current_demo = run_cv_simulation(demo_C, demo_D, demo_etai, demo_etaf, demo_sweep_rate, demo_n, demo_alpha, demo_k0, chem_rate, demo_T)
                
                # Calculate parameters
                tk_demo = 2*(demo_etai-demo_etaf)/demo_sweep_rate
                ktk_demo = chem_rate*tk_demo
                
                # Create plot
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(eta_demo, current_demo)
                ax.set_xlabel('Potential (V)')
                ax.set_ylabel('Current density (mA/cm²)')
                ax.set_title(f'EC Mechanism (kc = {chem_rate} s⁻¹, k·tₖ = {ktk_demo:.2f})')
                ax.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig)
                
                # Explanation
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
                
                # Create interactive demo
                et_rate = st.slider("Electron transfer rate constant (cm/s)", min_value=1e-5, max_value=1e-1, value=1e-3, format="%.1e")
                
                # Fixed parameters
                demo_C = 1.0
                demo_D = 1e-5
                demo_etai = 0.5
                demo_etaf = -0.5
                demo_sweep_rate = 0.05
                demo_n = 1.0
                demo_alpha = 0.5
                demo_kc = 0.01
                demo_T = 298.15
                
                # Run simulation
                eta_demo, current_demo = run_cv_simulation(demo_C, demo_D, demo_etai, demo_etaf, demo_sweep_rate, demo_n, demo_alpha, et_rate, demo_kc, demo_T)
                
                # Calculate parameters
                F = 96485
                R = 8.3145
                f = F/(R*demo_T)
                Lambda = et_rate/(demo_D*f*demo_sweep_rate)**0.5
                
                # Create plot
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(eta_demo, current_demo)
                ax.set_xlabel('Potential (V)')
                ax.set_ylabel('Current density (mA/cm²)')
                ax.set_title(f'Effect of k₀ = {et_rate:.1e} cm/s (Λ = {Lambda:.2f})')
                ax.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig)
                
                # Explanation
                reversibility = "Reversible" if Lambda > 15 else "Quasi-reversible" if Lambda > 1e-3 else "Irreversible"
                
                st.markdown(f"""
                **Observations:**
                
                - At this electron transfer rate ({et_rate:.1e} cm/s), the dimensionless parameter Λ = {Lambda:.2f}
                - The system is classified as: **{reversibility}**
                - {'Higher k₀ values lead to more reversible behavior with smaller peak separation' if et_rate > 1e-2 else 'Lower k₀ values lead to irreversible behavior with larger peak separation'}
                - {'The peaks are nearly symmetric and the peak separation approaches 59 mV/n' if Lambda > 15 else ''}
                """)
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
            
            # Create example CV waveform
            fig, ax = plt.subplots(figsize=(6, 3))
            time = np.linspace(0, 10, 1000)
            potential = np.concatenate([
                np.linspace(0, 0.5, 250),
                np.linspace(0.5, -0.5, 500),
                np.linspace(-0.5, 0, 250)
            ])
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
            
            # Create comparison of reversible vs EC voltammograms
            fig, ax = plt.subplots(figsize=(7, 4))
            
            # Generate example data
            potential = np.linspace(0.5, -0.5, 1000)
            
            # Simulate reversible response (no chemical reaction)
            
            current_rev = -10 * (np.exp(-40*(potential-0.05)) - np.exp(-40*(potential+0.05))) / (1 + np.exp(-40*(potential-0.05)) + np.exp(-40*(potential+0.05)))
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

def calculate_peak_data(eta, current):
    """Calculate key parameters from CV peaks"""
    # Find peaks
    max_current_idx = np.argmax(current)
    min_current_idx = np.argmin(current)
    
    # Calculate peak separation
    peak_separation = abs(eta[max_current_idx] - eta[min_current_idx])
    
    # Calculate peak current ratio
    peak_ratio = abs(current[min_current_idx]/current[max_current_idx])
    
    # For Randles-Sevcik analysis
    E_half = (eta[max_current_idx] + eta[min_current_idx]) / 2
    
    return {
        'max_idx': max_current_idx,
        'min_idx': min_current_idx,
        'max_potential': eta[max_current_idx],
        'min_potential': eta[min_current_idx],
        'max_current': current[max_current_idx],
        'min_current': current[min_current_idx],
        'peak_separation': peak_separation,
        'peak_ratio': peak_ratio,
        'E_half': E_half
    }

def presets_menu():
    """Create a dropdown menu with preset CV experiment types"""
    preset = st.selectbox(
        "Select experiment preset:",
        [
            "Custom settings",
            "Reversible system (ferrocene-like)",
            "Quasi-reversible metal complex",
            "Irreversible organic reduction",
            "Fast EC mechanism",
            "Slow EC mechanism"
        ]
    )
    
    if preset == "Reversible system (ferrocene-like)":
        return {
            'C': 1.0,
            'D': 7.2e-6,
            'etai': 0.4,
            'etaf': -0.4,
            'v': 0.1,
            'n': 1.0,
            'alpha': 0.5,
            'k0': 1e-1,
            'kc': 1e-6,
            'T': 298.15
        }
    elif preset == "Quasi-reversible metal complex":
        return {
            'C': 2.0,
            'D': 5.5e-6,
            'etai': 0.5,
            'etaf': -0.5,
            'v': 0.05,
            'n': 1.0,
            'alpha': 0.5,
            'k0': 5e-3,
            'kc': 1e-5,
            'T': 298.15
        }
    elif preset == "Irreversible organic reduction":
        return {
            'C': 0.5,
            'D': 9.0e-6,
            'etai': 0.3,
            'etaf': -0.7,
            'v': 0.2,
            'n': 2.0,
            'alpha': 0.3,
            'k0': 1e-4,
            'kc': 1e-6,
            'T': 298.15
        }
    elif preset == "Fast EC mechanism":
        return {
            'C': 1.0, 
            'D': 8.0e-6,
            'etai': 0.5,
            'etaf': -0.5,
            'v': 0.1,
            'n': 1.0,
            'alpha': 0.5,
            'k0': 1e-2,
            'kc': 10.0,
            'T': 298.15
        }
    elif preset == "Slow EC mechanism":
        return {
            'C': 1.0,
            'D': 8.0e-6,
            'etai': 0.5,
            'etaf': -0.5,
            'v': 0.1,
            'n': 1.0,
            'alpha': 0.5,
            'k0': 1e-2,
            'kc': 0.1,
            'T': 298.15
        }
    else:  # Custom settings
        return None

def create_streamlit_app():
    st.set_page_config(page_title="Cyclic Voltammetry Simulator", layout="centered")
    
    st.title("Cyclic Voltammetry Simulation")
    st.write("Based on Bard and Faulkner, Appendix B - EC mechanism")
    
    # Tabs for different modes
    tab1, tab2, tab3, tab4 = st.tabs(["Simulator", "Parameter Study", "Compare Models", "Educational"])
    
    with tab1:
        # Model selection
        model_type = st.radio(
            "Select simulation model:",
            ["Standard (Current Density)", "Dimensionless"],
            horizontal=True
        )
        
        # Preset selection
        preset_values = presets_menu()
        
        # Create two columns for the layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Simulation Parameters")
            
            # Independent variables with sliders and input boxes
            C = st.number_input("Concentration [mol/L]", 
                               min_value=0.01, max_value=10.0, 
                               value=preset_values['C'] if preset_values else 1.0, 
                               step=0.1)
            
            D = st.number_input("Diffusion coef. [cm²/s]", 
                               min_value=1e-6, max_value=1e-4, 
                               value=preset_values['D'] if preset_values else 1e-5, 
                               format="%.1e", step=1e-6)
            
            # Potential settings
            st.subheader("Potential Settings")
            etai = st.slider("Initial E [V]", 
                            min_value=-1.0, max_value=1.0, 
                            value=preset_values['etai'] if preset_values else 0.2, 
                            step=0.1)
            
            etaf = st.slider("Final E [V]", 
                            min_value=-1.0, max_value=1.0, 
                            value=preset_values['etaf'] if preset_values else -0.2, 
                            step=0.1)
            
            v = st.number_input("Sweep rate [V/s]", 
                               min_value=1e-4, max_value=1.0, 
                               value=preset_values['v'] if preset_values else 1e-3, 
                               format="%.1e", step=1e-4)
            
            # Advanced parameters (collapsible)
            with st.expander("Advanced Parameters"):
                n = st.number_input("Electrons transferred", 
                                   min_value=0.1, max_value=10.0, 
                                   value=preset_values['n'] if preset_values else 1.0, 
                                   step=0.1)
                
                alpha = st.number_input("Transfer coefficient", 
                                       min_value=0.1, max_value=0.9, 
                                       value=preset_values['alpha'] if preset_values else 0.5, 
                                       step=0.1)
                
                k0 = st.number_input("Rate constant [cm/s]", 
                                    min_value=1e-5, max_value=1.0, 
                                    value=preset_values['k0'] if preset_values else 1e-2, 
                                    format="%.1e", step=1e-3)
                
                kc = st.number_input("Chemical rate [1/s]", 
                                    min_value=1e-5, max_value=10.0, 
                                    value=preset_values['kc'] if preset_values else 1e-3, 
                                    format="%.1e", step=1e-4)
                
                T = st.number_input("Temperature [K]", 
                                   min_value=273.15, max_value=373.15, 
                                   value=preset_values['T'] if preset_values else 298.15, 
                                   step=5.0)
        
        with col2:
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
            
            # Classification of electrochemical reversibility
            if Lambda > 15:
                reversibility = "Reversible"
                rev_color = "green"
            elif Lambda > 1e-3:
                reversibility = "Quasi-reversible"
                rev_color = "orange"
            else:
                reversibility = "Irreversible"
                rev_color = "red"
                
            # Classification of chemical reaction effect
            if ktk > 10:
                ec_type = "Strong EC effect"
                ec_color = "red"
            elif ktk > 0.1:
                ec_type = "Moderate EC effect"
                ec_color = "orange"
            else:
                ec_type = "Weak EC effect"
                ec_color = "green"
                
            # Display key parameters
            param_col1, param_col2, param_col3 = st.columns(3)
            with param_col1:
                st.metric("Kinetic param (k·tₖ)", f"{ktk:.2e}", 
                          delta=ec_type, delta_color="off")
            with param_col2:
                st.metric("Norm. kinetic (k·tₖ/L)", f"{km:.2e}", 
                          delta="Warning!" if km > 0.1 else "OK", 
                          delta_color="off" if km <= 0.1 else "red")
            with param_col3:
                st.metric("Reversibility (Λ)", f"{Lambda:.2e}", 
                          delta=reversibility, delta_color="off")
            
            # Plot the results
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(eta, current)
            ax.set_xlabel('Overpotential (V)')
            ax.set_ylabel(y_label)
            ax.set_title(title)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Get peak data
            peak_data = calculate_peak_data(eta, current)
            
            # Plot the points
            ax.plot(peak_data['max_potential'], peak_data['max_current'], 'ro')
            ax.plot(peak_data['min_potential'], peak_data['min_current'], 'ro')
            
            # Add compact labels directly on the plot
            max_label = f"({peak_data['max_potential']:.2f}V, {peak_data['max_current']:.2f})"
            min_label = f"({peak_data['min_potential']:.2f}V, {peak_data['min_current']:.2f})"
            
            # Calculate label positions
            y_range = max(current) - min(current)
            
            # For the maximum peak (typically at the top of the plot)
            if peak_data['max_current'] > 0:
                # If in upper half, place label below the point
                y_offset = -y_range * 0.05
                va = 'top'
            else:
                # If in lower half, place label above the point
                y_offset = y_range * 0.05
                va = 'bottom'
                
            ax.annotate(max_label, 
                        xy=(peak_data['max_potential'], peak_data['max_current']),
                        xytext=(peak_data['max_potential'], peak_data['max_current'] + y_offset),
                        ha='center', va=va,
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='gray'))
            
            # For the minimum peak (typically at the bottom of the plot)
            if peak_data['min_current'] < 0:
                # If in lower half, place label above the point
                y_offset = y_range * 0.05
                va = 'bottom'
            else:
                # If in upper half, place label below the point
                y_offset = -y_range * 0.05
                va = 'top'
                
            ax.annotate(min_label, 
                        xy=(peak_data['min_potential'], peak_data['min_current']),
                        xytext=(peak_data['min_potential'], peak_data['min_current'] + y_offset),
                        ha='center', va=va,
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='gray'))
            
            # Make sure plot has enough margins
            fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
            st.pyplot(fig)
            
            # Extra analysis information
            with st.expander("Peak Analysis"):
                # Create two columns for the analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Forward Peak (Maximum)**")
                    st.markdown(f"- Potential: {peak_data['max_potential']:.4f} V")
                    st.markdown(f"- Current: {peak_data['max_current']:.4f} {y_label.split('(')[0].strip()}")
                
                with col2:
                    st.markdown("**Reverse Peak (Minimum)**")
                    st.markdown(f"- Potential: {peak_data['min_potential']:.4f} V")
                    st.markdown(f"- Current: {peak_data['min_current']:.4f} {y_label.split('(')[0].strip()}")
                
                st.markdown("**Diagnostic Parameters**")
                
                # Show peak ratio - key diagnostic for EC mechanism
                peak_ratio_col, peak_sep_col = st.columns(2)
                with peak_ratio_col:
                    st.metric("Peak Current Ratio (|iᵣₑᵥ/iₒₓ|)", 
                              f"{peak_data['peak_ratio']:.4f}",
                              delta="Reversible" if 0.9 < peak_data['peak_ratio'] < 1.1 else 
                                   "EC mechanism" if peak_data['peak_ratio'] < 0.9 else 
                                   "Other effects",
                              delta_color="off")
                
                # For reversible system, peak separation should be 59/n mV
                with peak_sep_col:
                    theoretical_sep = 0.059/n
                    st.metric("Peak Separation", 
                              f"{peak_data['peak_separation']:.4f} V",
                              delta=f"Theo: {theoretical_sep:.3f} V",
                              delta_color="off")
                
                # Expected response based on diagnostics
                st.markdown("**Electrochemical System Diagnosis**")
                
                if 0.9 < peak_data['peak_ratio'] < 1.1 and abs(peak_data['peak_separation'] - 0.059/n) < 0.015:
                    st.success("This appears to be a reversible system with no significant chemical reaction.")
                elif peak_data['peak_ratio'] < 0.9 and abs(peak_data['peak_separation'] - 0.059/n) < 0.03:
                    st.warning(f"This appears to be an EC mechanism - chemical reaction is consuming species R.")
                elif peak_data['peak_separation'] > 0.090/n:
                    st.warning(f"This appears to be a quasi-reversible or irreversible system due to slow electron transfer.")
                else:
                    st.info("This system shows mixed kinetic effects.")
                
                # Randles-Sevcik equation
                if model_type == "Standard (Current Density)" and n and D:
                    st.markdown("**Randles-Sevcik Analysis**")
                    st.markdown(f"Theoretical peak current: {0.4463*n*F*C*(n*F*v*D/(R*T))**0.5:.4f} mA/cm²")
                    st.markdown(f"Observed peak current: {peak_data['max_current']:.4f} mA/cm²")
            
            # Download options
            csv_data = np.column_stack((eta, current))
            st.download_button(
                label="Download Data as CSV",
                data=convert_to_csv(csv_data, header),
                file_name=file_name,
                mime="text/csv"
            )

    
    with tab2:
        st.subheader("Parameter Study")
        
        # Select parameter to study
        study_param = st.selectbox(
            "Select parameter to study:",
            ["Sweep rate (v)", "Concentration (C)", "Rate constant (k₀)", "Chemical rate (kc)"]
        )
        
        # Select model
        model_study = st.radio(
            "Select simulation model:",
            ["Standard (Current Density)", "Dimensionless"],
            horizontal=True
        )
        
        # Create parameter ranges
        if study_param == "Sweep rate (v)":
            param_min = st.number_input("Minimum sweep rate [V/s]", min_value=1e-4, max_value=0.1, value=1e-3, format="%.1e")
            param_max = st.number_input("Maximum sweep rate [V/s]", min_value=1e-3, max_value=1.0, value=5e-2, format="%.1e")
            param_name = "v"
            param_unit = "V/s"
            param_values = np.logspace(np.log10(param_min), np.log10(param_max), 5)
            
        elif study_param == "Concentration (C)":
            param_min = st.number_input("Minimum concentration [mol/L]", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
            param_max = st.number_input("Maximum concentration [mol/L]", min_value=1.0, max_value=10.0, value=5.0, step=0.5)
            param_name = "C"
            param_unit = "mol/L"
            param_values = np.linspace(param_min, param_max, 5)
            
        elif study_param == "Rate constant (k₀)":
            param_min = st.number_input("Minimum rate constant [cm/s]", min_value=1e-4, max_value=1e-2, value=1e-3, format="%.1e")
            param_max = st.number_input("Maximum rate constant [cm/s]", min_value=1e-2, max_value=1.0, value=1e-1, format="%.1e")
            param_name = "k0"
            param_unit = "cm/s"
            param_values = np.logspace(np.log10(param_min), np.log10(param_max), 5)
            
        else:  # Chemical rate
            param_min = st.number_input("Minimum chemical rate [1/s]", min_value=1e-4, max_value=1e-2, value=1e-3, format="%.1e")
            param_max = st.number_input("Maximum chemical rate [1/s]", min_value=1e-2, max_value=10.0, value=1e-1, format="%.1e")
            param_name = "kc"
            param_unit = "1/s"
            param_values = np.logspace(np.log10(param_min), np.log10(param_max), 5)
        
        # Fixed parameters
        with st.expander("Fixed Parameters"):
            if param_name != "C":
                C = st.number_input("Concentration [mol/L]", min_value=0.01, max_value=10.0, value=1.0, step=0.1, key="study_C")
            if param_name != "v":
                v = st.number_input("Sweep rate [V/s]", min_value=1e-4, max_value=1.0, value=1e-3, format="%.1e", key="study_v")
            if param_name != "k0":
                k0 = st.number_input("Rate constant [cm/s]", min_value=1e-5, max_value=1.0, value=1e-2, format="%.1e", key="study_k0")
            if param_name != "kc":
                kc = st.number_input("Chemical rate [1/s]", min_value=1e-5, max_value=1.0, value=1e-3, format="%.1e", key="study_kc")
                
            D = st.number_input("Diffusion coef. [cm²/s]", min_value=1e-6, max_value=1e-4, value=1e-5, format="%.1e", key="study_D")
            etai = st.slider("Initial E [V]", min_value=-1.0, max_value=1.0, value=0.2, step=0.1, key="study_etai")
            etaf = st.slider("Final E [V]", min_value=-1.0, max_value=1.0, value=-0.2, step=0.1, key="study_etaf")
            n = st.number_input("Electrons transferred", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key="study_n")
            alpha = st.number_input("Transfer coefficient", min_value=0.1, max_value=0.9, value=0.5, step=0.1, key="study_alpha")
            T = st.number_input("Temperature [K]", min_value=273.15, max_value=373.15, value=298.15, step=5.0, key="study_T")
        
        # Run simulations for multiple parameter values
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Create a color map
        colors = plt.cm.viridis(np.linspace(0, 1, len(param_values)))
        
        for i, param_val in enumerate(param_values):
            # Set the parameter value
            if param_name == "C":
                C = param_val
            elif param_name == "v":
                v = param_val
            elif param_name == "k0":
                k0 = param_val
            elif param_name == "kc":
                kc = param_val
            
            # Run simulation
            if model_study == "Standard (Current Density)":
                eta, current = run_cv_simulation(C, D, etai, etaf, v, n, alpha, k0, kc, T)
                y_label = "Current density (mA/cm²)"
            else:
                eta, current = run_dimensionless_cv_simulation(C, D, etai, etaf, v, n, alpha, k0, kc, T)
                y_label = "Dimensionless current"
            
            # Plot the results with different colors
            ax.plot(eta, current, color=colors[i], label=f"{param_val:.3g} {param_unit}")
        
        ax.set_xlabel('Overpotential (V)')
        ax.set_ylabel(y_label)
        ax.set_title(f'Effect of {study_param}')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(title=f"{study_param}")
        
        fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
        st.pyplot(fig)
        
        st.markdown("**Insights:**")
        
        # Provide insights based on the parameter study
        if study_param == "Sweep rate (v)":
            st.markdown("""
            - Faster sweep rates lead to larger peak currents
            - Peak separation increases with increasing sweep rate
            - At very fast sweep rates, the system becomes more irreversible
            """)
        elif study_param == "Concentration (C)":
            st.markdown("""
            - Peak current is proportional to concentration
            - Peak positions remain relatively unchanged with concentration
            - Higher concentrations lead to larger diffusion-limited currents
            """)
        elif study_param == "Rate constant (k₀)":
            st.markdown("""
            - Low k₀ values lead to irreversible behavior (larger peak separation)
            - High k₀ values approach reversible behavior (smaller peak separation)
            - The transition from irreversible to reversible typically occurs around k₀ = 0.01 cm/s
            """)
        else:  # Chemical rate
            st.markdown("""
            - Higher chemical rate constants reduce the reverse peak height
            - This reflects how the chemical reaction consumes species R after reduction
            - Very high kc values may completely eliminate the reverse peak (irreversible EC mechanism)
            """)
    
    with tab3:
        st.subheader("Compare Standard vs. Dimensionless Models")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Simulation Parameters")
            
            # Independent variables with sliders and input boxes
            C_comp = st.number_input("Concentration [mol/L]", min_value=0.01, max_value=10.0, value=1.0, step=0.1, key="comp_C")
            D_comp = st.number_input("Diffusion coef. [cm²/s]", min_value=1e-6, max_value=1e-4, value=1e-5, format="%.1e", step=1e-6, key="comp_D")
            etai_comp = st.slider("Initial E [V]", min_value=-1.0, max_value=1.0, value=0.2, step=0.1, key="comp_etai")
            etaf_comp = st.slider("Final E [V]", min_value=-1.0, max_value=1.0, value=-0.2, step=0.1, key="comp_etaf")
            v_comp = st.number_input("Sweep rate [V/s]", min_value=1e-4, max_value=1.0, value=1e-3, format="%.1e", step=1e-4, key="comp_v")
        
        with col2:
            st.markdown("### Advanced Parameters")
            # Advanced parameters
            n_comp = st.number_input("Electrons transferred", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key="comp_n")
            alpha_comp = st.number_input("Transfer coefficient", min_value=0.1, max_value=0.9, value=0.5, step=0.1, key="comp_alpha")
            k0_comp = st.number_input("Rate constant [cm/s]", min_value=1e-5, max_value=1.0, value=1e-2, format="%.1e", step=1e-3, key="comp_k0")
            kc_comp = st.number_input("Chemical rate [1/s]", min_value=1e-5, max_value=1.0, value=1e-3, format="%.1e", step=1e-4, key="comp_kc")
            T_comp = st.number_input("Temperature [K]", min_value=273.15, max_value=373.15, value=298.15, step=5.0, key="comp_T")
            
        # Run both simulations
        eta_std, current_std = run_cv_simulation(C_comp, D_comp, etai_comp, etaf_comp, v_comp, n_comp, alpha_comp, k0_comp, kc_comp, T_comp)
        eta_dim, current_dim = run_dimensionless_cv_simulation(C_comp, D_comp, etai_comp, etaf_comp, v_comp, n_comp, alpha_comp, k0_comp, kc_comp, T_comp)
        
        # Scale dimensionless current to match standard model
        # This is for visualization purposes to compare shapes
        scale_factor = st.slider("Scale dimensionless current by:", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        current_dim_scaled = current_dim * scale_factor
        
        # Plot the results
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(eta_std, current_std, 'b-', label='Standard Model')
        ax.plot(eta_dim, current_dim_scaled, 'r--', label='Dimensionless Model')
        
        ax.set_xlabel('Overpotential (V)')
        ax.set_ylabel('Current')
        ax.set_title('Model Comparison')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
        st.pyplot(fig)
        
        st.markdown("""
        ### Key Differences Between Models
        
        1. **Standard Model:**
           - Uses physical units for current density (mA/cm²)
           - More intuitive for direct comparison with experimental data
           - Rate constants have physical units (cm/s)
        
        2. **Dimensionless Model:**
           - Uses dimensionless variables as in Bard & Faulkner Appendix B
           - Enables more general analysis independent of specific parameters
           - Rate constants are in dimensionless form
        
        Both models describe the same electrochemical system but with different mathematical approaches.
        """)
        
    # Add explanation and references (shown on all tabs)
    with st.expander("About this simulation"):
        st.markdown("""
        ### Cyclic Voltammetry Simulation Details
        
        This application simulates cyclic voltammetry experiments for an EC mechanism (electrochemical reaction followed by a chemical reaction) based on the digital simulation approach described in Bard and Faulkner's "Electrochemical Methods: Fundamentals and Applications," Appendix B.
        
        #### Electrochemical Mechanism:
        
        The EC mechanism consists of:
        1. An electron transfer step: O + e⁻ ⟷ R (electrochemical)
        2. A following chemical reaction: R → Z (chemical)
        
        #### Key Parameters:
        
        - **Kinetic parameter (k·tₖ)**: Indicates the importance of the following chemical reaction
        - **Normalized kinetic parameter (k·tₖ/L)**: Should be less than 0.1 for accurate simulation
        - **Reversibility parameter (Λ)**: Indicates the electrochemical reversibility
          - Λ > 15: Reversible
          - 15 > Λ > 10⁻³: Quasi-reversible
          - Λ < 10⁻³: Irreversible
        
        #### Reference:
        Bard, A.J.; Faulkner, L.R. Electrochemical Methods: Fundamentals and Applications, 2nd ed.; Wiley: New York, 2001.
        
        This simulation was converted from MATLAB to Python by Claude AI.
        """)
# Make sure plot has enough margins
    fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
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
