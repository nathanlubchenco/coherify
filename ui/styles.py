"""
Professional styling and color schemes for the Coherify UI.
"""

# Professional color palette (colorblind-aware)
COLORS = {
    # Primary palette - light blues, greens, white
    "primary_blue": "#2E86AB",      # Deep blue
    "secondary_blue": "#A2D2FF",    # Light blue  
    "primary_green": "#06A77D",     # Teal green
    "secondary_green": "#B7E4C7",   # Light green
    "accent_gray": "#6C757D",       # Professional gray
    "light_gray": "#F8F9FA",        # Very light gray
    "white": "#FFFFFF",             # Pure white
    
    # Chart colors (colorblind-aware - using viridis-inspired palette)
    "chart_colors": [
        "#2E86AB",  # Blue
        "#06A77D",  # Teal
        "#F18F01",  # Orange
        "#A23B72",  # Purple
        "#6C757D",  # Gray
    ],
    
    # Status colors
    "success": "#06A77D",
    "warning": "#F18F01", 
    "error": "#D62728",
    "info": "#2E86AB"
}

# Professional CSS styling
PROFESSIONAL_CSS = """
<style>
    /* Import professional font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styling */
    .main {
        background-color: #FFFFFF;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.25rem;
        font-weight: 600;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem 0;
        border-bottom: 2px solid #A2D2FF;
        background: linear-gradient(135deg, #F8F9FA 0%, #FFFFFF 100%);
        border-radius: 8px;
    }
    
    .section-header {
        font-size: 1.25rem;
        font-weight: 500;
        color: #06A77D;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #B7E4C7;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8F9FA 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #A2D2FF;
        box-shadow: 0 2px 8px rgba(46, 134, 171, 0.1);
        margin: 0.5rem 0;
        transition: box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 16px rgba(46, 134, 171, 0.15);
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #F8F9FA 0%, #A2D2FF 100%);
        padding: 1.25rem;
        border-radius: 8px;
        border-left: 4px solid #2E86AB;
        margin: 1rem 0;
        color: #1a1a1a;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    .success-box {
        background: linear-gradient(135deg, #F8F9FA 0%, #B7E4C7 100%);
        border-left-color: #06A77D;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #FFF9E6 0%, #FFE4B5 100%);
        border-left-color: #F18F01;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #F8F9FA;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #2E86AB 0%, #06A77D 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(46, 134, 171, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(46, 134, 171, 0.3);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border: 1px solid #A2D2FF;
        border-radius: 6px;
        padding: 0.75rem;
        font-family: 'Inter', sans-serif;
        transition: border-color 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #2E86AB;
        box-shadow: 0 0 0 2px rgba(46, 134, 171, 0.1);
    }
    
    /* Metric styling */
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #A2D2FF;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 600;
        color: #2E86AB;
    }
    
    .metric-label {
        font-size: 0.875rem;
        font-weight: 500;
        color: #6C757D;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #2E86AB 0%, #06A77D 100%);
    }
    
    /* Remove default margins */
    .element-container {
        margin-bottom: 0.5rem !important;
    }
    
    /* Clean selectbox styling */
    .stSelectbox > div > div {
        border: 1px solid #A2D2FF;
        border-radius: 6px;
    }
    
    /* Checkbox styling */
    .stCheckbox > label {
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        color: #1a1a1a;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #F8F9FA;
        border-radius: 8px;
        padding: 0.25rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 6px;
        color: #6C757D;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2E86AB;
        color: white;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #F8F9FA;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #A2D2FF;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #2E86AB;
    }
</style>
"""

def get_chart_config():
    """Get professional chart configuration."""
    return {
        "colors": COLORS["chart_colors"],
        "background": COLORS["white"],
        "grid_color": COLORS["light_gray"],
        "text_color": "#1a1a1a",
        "font_family": "Inter",
        "font_size": 12,
        "title_font_size": 16,
        "height": 400,
    }

def apply_professional_styling():
    """Apply professional CSS styling to the Streamlit app."""
    import streamlit as st
    st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)