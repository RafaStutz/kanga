import numpy as np
import pandas as pd
from pathlib import Path
import json

from sklearn.preprocessing import StandardScaler, MinMaxScaler

def _signed_log10(arr: np.ndarray, low: float = 1e-30) -> np.ndarray:
    """log10 mantendo o sinal – bom p/ variáveis físicas que cruzam 0."""
    sign = np.sign(arr)
    return sign * np.log10(np.clip(np.abs(arr), low, None))

def normalize_df(
    df: pd.DataFrame,
    *,
    target_col: str | None = "target",
    how: str = "standard",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Normaliza todas as colunas numéricas do DataFrame.

    Parameters
    ----------
    df : DataFrame de entrada
    target_col : coluna a **preservar** (não é normalizada). Use None se não houver.
    how : {"standard", "minmax", "log_standard"}
    inplace : se True, modifica `df`; senão devolve uma cópia

    Returns
    -------
    DataFrame normalizado (se inplace=False, é uma cópia)
    """
    if not inplace:
        df = df.copy()

    numeric_cols = df.select_dtypes(include=["float", "int"]).columns
    if target_col and target_col in numeric_cols:
        numeric_cols = numeric_cols.drop(target_col)

    if how == "standard":
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    elif how == "minmax":
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    elif how == "log_standard":
        # 1) signed-log transform
        df[numeric_cols] = _signed_log10(df[numeric_cols].values)
        # 2) z-score
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    else:
        raise ValueError(f"`how` inválido: {how}")

    return df

class FeynmanDataFrameGenerator:
    """
    Generates 15 separate DataFrames, one for each equation from the images.
    Each DataFrame contains 10,000 samples.
    Tabular format: [var1, var2, ..., target]
    """
    
    def __init__(self):
        self.equations = self._define_equations()
    
    def _define_equations(self) -> dict:
        """Defines 15 feynman equations."""

        equations = {}
        
        # ============ EASY SET ============
        
        # I.12.1: F = mu * N_n
        equations["I_12_1"] = {
            "formula": "F = mu * N",
            "variables": ["mu", "N"],
            "sampling": {
                "mu": ("uniform", 1, 5),            
                "N": ("uniform", 1, 5)              
            },
            "function": lambda mu, N: mu * N
        }
        
        # I.12.4: E = q1/(4*pi*epsilon*r^2)
        equations["I_12_4"] = {
            "formula": "E = q/(4*pi*epsilon*r^2)",
            "variables": ["q", "r"],
            "sampling": {
                "q": ("uniform", 1, 5),             
                "r": ("uniform", 1, 5)              
            },
            "function": lambda q, r: q / (4 * np.pi * 8.854e-12 * r**2)
        }
        
        # I.12.5: F = q2 * E
        equations["I_12_5"] = {
            "formula": "F = q * E",
            "variables": ["q", "E"],
            "sampling": {
                "q": ("uniform", 1, 5),              
                "E": ("uniform", 1, 5)               
            },
            "function": lambda q, E: q * E
        }
        
        # I.14.3: U = mgh
        equations["I_14_3"] = {
            "formula": "U = m*g*h",
            "variables": ["m", "h"],
            "sampling": {
                "m": ("uniform", 1, 5),            
                "h": ("uniform", 1, 5)             
            },
            "function": lambda m, h: m * 9.807 * h
        }
        
        # I.14.4: U = (1/2)*k_spring*x^2
        equations["I_14_4"] = {
            "formula": "U = 0.5*k*x^2",
            "variables": ["k", "x"],
            "sampling": {
                "k": ("uniform", 1, 5),                 
                "x": ("uniform", 1, 5)                 
            },
            "function": lambda k, x: 0.5 * k * x**2
        }
        
        # ============ MEDIUM SET ============
        
        # I.8.14: d = sqrt((x2-x1)^2 + (y2-y1)^2)
        equations["I_8_14"] = {
            "formula": "d = sqrt((x2-x1)^2 + (y2-y1)^2)",
            "variables": ["x1", "x2", "y1", "y2"],
            "sampling": {
                "x1": ("uniform", 1, 5),              
                "x2": ("uniform", 1, 5),              
                "y1": ("uniform", 1, 5),              
                "y2": ("uniform", 1, 5)               
            },
            "function": lambda x1, x2, y1, y2: np.sqrt((x2-x1)**2 + (y2-y1)**2)
        }
        
        # I.10.7: m = m0/sqrt(1-v^2/c^2)
        equations["I_10_7"] = {
            "formula": "m = m0/sqrt(1-v^2/c^2)",
            "variables": ["m0", "v"],
            "sampling": {
                "m0": ("uniform", 1, 5),       
                "v": ("uniform", 1, 2)       
            },
            "function": lambda m0, v: m0 / np.sqrt(1 - (v/2.998e8)**2)
        }
        
        # I.11.19: A = x1*y1 + x2*y2 + x3*y3
        equations["I_11_19"] = {
            "formula": "A = x1*y1 + x2*y2 + x3*y3",
            "variables": ["x1", "x2", "x3", "y1", "y2", "y3"],
            "sampling": {
                "x1": ("uniform", 1, 5),         
                "x2": ("uniform", 1, 5),         
                "x3": ("uniform", 1, 5),         
                "y1": ("uniform", 1, 5),         
                "y2": ("uniform", 1, 5),         
                "y3": ("uniform", 1, 5)         
            },
            "function": lambda x1, x2, x3, y1, y2, y3: x1*y1 + x2*y2 + x3*y3
        }
        
        # I.12.2: F = q1*q2/(4*pi*epsilon*r^2)
        equations["I_12_2"] = {
            "formula": "F = q1*q2/(4*pi*epsilon*r^2)",
            "variables": ["q1", "q2", "r"],
            "sampling": {
                "q1": ("uniform", 1, 5) ,              
                "q2": ("uniform", 1, 5) ,              
                "r": ("uniform", 1, 5)                 
            },
            "function": lambda q1, q2, r: q1*q2 / (4 * np.pi * 8.854e-12 * r**2)
        }
        
        # I.12.11: F = q*(E + B*v*sin(theta))
        equations["I_12_11"] = {
            "formula": "F = q*(E + B*v*sin(theta))",
            "variables": ["q", "E", "B", "v", "theta"],
            "sampling": {
                "q": ("uniform", 1, 5) ,                
                "E": ("uniform", 1, 5) ,                
                "B": ("uniform", 1, 5) ,                
                "v": ("uniform", 1, 5) ,                
                "theta": ("uniform", 1, 5)          
            },
            "function": lambda q, E, B, v, theta: q * (E + B*v*np.sin(theta))
        }
        
        # ============ HARD SET ============
        
        # B2
        equations["B2"] = {
            "formula": "k = mkG/L^2 * (1 + sqrt(1 + 2EL^2/(mk^2G)) * cos(theta1 - theta2))",
            "variables": ["m", "k_G", "L", "E", "theta1", "theta2"],
            "sampling": {
                "m": ("uniform", 1, 3),               
                "k_G": ("uniform", 1, 3),              
                "L": ("uniform", 1, 3),                
                "E": ("uniform", 1, 3),               
                "theta1": ("uniform", 0, 6),
                "theta2": ("uniform", 0, 6) 
            },
            "function": lambda m, k_G, L, E, theta1, theta2: 
                m*k_G/(L**2) * (1 + np.sqrt(1 + 2*E*L**2/(m*k_G**2)) * np.cos(theta1 - theta2))
        }
        
        # B3
        equations["B3"] = {
            "formula": "r = d*(1-alpha^2)/(1+alpha*cos(theta1-theta2))",
            "variables": ["d", "alpha", "theta1", "theta2"],
            "sampling": {
                "d": ("uniform", 1, 3),                
                "alpha": ("uniform", 2, 4),     
                "theta1": ("uniform", 4, 5),
                "theta2": ("uniform", 4, 5) 
            },
            "function": lambda d, alpha, theta1, theta2: 
                d * (1 - alpha**2) / (1 + alpha * np.cos(theta1 - theta2))
        }
        
        # B4
        equations["B4"] = {
            "formula": "v = sqrt(2/m * (E - U - L^2/(2*m*r^2)))",
            "variables": ["m", "E", "U", "L", "r"],
            "sampling": {
                "m": ("uniform", 1, 3),               
                "E": ("uniform", 8, 12),            
                "U": ("uniform", 1, 3),             
                "L": ("uniform", 1, 3),             
                "r": ("uniform", 1, 3)              
            },
            "function": lambda m, E, U, L, r: np.sqrt(np.abs(2/m * (E - U - L**2/(2*m*r**2))))
        }
        
        # B5
        equations["B5"] = {
            "formula": "t = 2*pi*d^(3/2)/sqrt(G*(m1+m2))",
            "variables": ["d", "m1", "m2"],
            "sampling": {
                "d": ("uniform", 1, 3),                
                "m1": ("uniform", 1, 3),              
                "m2": ("uniform", 1, 3)               
            },
            "function": lambda d, m1, m2: 2 * np.pi * d**(3/2) / np.sqrt(6.674e-11 * (m1 + m2))
        }
        
        # B6
        equations["B6"] = {
            "formula": "alpha = sqrt(1 + 2*epsilon^2*E*L^2/(m*(Z1*Z2*q**2)^2))",
            "variables": ["epsilon", "E", "L", "m", "Z1", "Z2", "q"],
            "sampling": {
                "epsilon": ("uniform", 1, 3),      
                "E": ("uniform", 1, 3),            
                "L": ("uniform", 1, 3),              
                "m": ("uniform", 1, 3),            
                "Z1": ("uniform", 1, 3),                 
                "Z2": ("uniform", 1, 3),                
                "q": ("uniform", 1, 3)               
            },
            "function": lambda epsilon, E, L, m, Z1, Z2, q: 
                np.sqrt(1 + 2*epsilon**2*E*L**2/(m*(Z1*Z2*q**2)**2))
        }
        
        return equations
    
    def _sample_variable(self, dist_type: str, param1: float, param2: float, n_samples: int) -> np.ndarray:
        """Generates samples according to the specifications."""
        if dist_type == "uniform":
            return np.random.uniform(param1, param2, n_samples)
        elif dist_type == "log_uniform":
            log_samples = np.random.uniform(param1, param2, n_samples)
            return 10 ** log_samples
        else:
            raise ValueError(f"Tipo '{dist_type}' não suportado")

    def generate_single_dataframe(self, equation_key: str, size: int = 1000) -> pd.DataFrame:
        """
        Generates a DataFrame for a specific equation.

        Returns:
            DataFrame with columns [var1, var2, ..., target]
        """
        eq = self.equations[equation_key]
        variables = eq["variables"]
        sampling = eq["sampling"]
        func = eq["function"]
        total_size = size
        
        print(f" Generating a dataframe for {equation_key}")
        print(f"    Formula: {eq['formula']}")
        print(f"    Variables: {variables}")

        samples = {}
        for var in variables:
            dist_type, param1, param2 = sampling[var]
            samples[var] = self._sample_variable(dist_type, param1, param2, total_size)
        
        try:
            target = func(*[samples[var] for var in variables])
            
            valid_mask = np.isfinite(target) & (target != 0) & (np.abs(target) > 1e-50)
            
            if np.sum(valid_mask) < total_size * 0.8:
                print(f"  Only {np.sum(valid_mask)}/{total_size} valid samples. Regenerating...")
                return self.generate_single_dataframe(equation_key, int(train_size * 1.5), int(test_size * 1.5))
            
            for var in variables:
                samples[var] = samples[var][valid_mask]
            target = target[valid_mask]
            
            if len(target) > total_size:
                indices = np.random.choice(len(target), total_size, replace=False)
                for var in variables:
                    samples[var] = samples[var][indices]
                target = target[indices]
            
            data_dict = samples.copy()
            data_dict['target'] = target
            
            df = pd.DataFrame(data_dict)

            print(f"   DataFrame created: {len(df)} samples")
            print(f"    Target range: [{target.min():.2e}, {target.max():.2e}]")

            
            return df
            
        except Exception as e:
            print(f"  Erro: {e}")
            return pd.DataFrame()

    def generate_all_dataframes(self, size: int = 1000) -> dict[str, pd.DataFrame]:
        """
        Generates all 15 DataFrames.
        
        Returns:
            dict with 15 DataFrames: {'I_12_1': df1, 'I_12_4': df2, ...}
        """
        dataframes = {}
        
        for i, (eq_key, eq_info) in enumerate(self.equations.items(), 1):

            df = self.generate_single_dataframe(eq_key, size)

            if not df.empty:
                dataframes[eq_key] = df
        
        if dataframes:
            first_key = list(dataframes.keys())[0]
            first_df = dataframes[first_key]
        
        return dataframes
    
    def save_dataframes(self, dataframes: dict[str, pd.DataFrame], output_dir: str = "./feynman_dataframes"):
        """Save DataFrames as CSV."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for eq_key, df in dataframes.items():
            csv_path = output_path / f"{eq_key}.csv"
            df.to_csv(csv_path, index=False)
        
        summary = {
            "total_dataframes": len(dataframes),
            "equations": {}
        }
        
        for eq_key, df in dataframes.items():
            eq_info = self.equations[eq_key]
            summary["equations"][eq_key] = {
                "formula": eq_info["formula"],
                "variables": eq_info["variables"],
                "total_samples": len(df),
                "target_mean": float(df['target'].mean()),
                "target_std": float(df['target'].std())
            }
        
        with open(output_path / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        


def generate_feynman_dataframes(size: int = 100, save: bool = True) -> dict[str, pd.DataFrame]:
    """
    Function to generate the 15 DataFrames feynman.
    
    Args:
        size: Number of samples per DataFrame
        test_size: Number of testing samples per DataFrame
        save: Whether to save the DataFrames as CSV
        
    Returns:
        dict with the 15 generated DataFrames
    """
    generator = FeynmanDataFrameGenerator()
    dataframes = generator.generate_all_dataframes(size)
    
    if save and dataframes:
        generator.save_dataframes(dataframes)
    
    return dataframes

if __name__ == "__main__":
    dfs = generate_feynman_dataframes(size=3000, save=True)

    if dfs:
        example_key = list(dfs.keys())[0]
        example_df = dfs[example_key]