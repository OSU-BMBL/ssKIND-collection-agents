
import re
import logging
import pandas as pd
from io import StringIO

logger = logging.getLogger(__name__)

def convert_html_table_to_dataframe(table: str):
    try:
        # remove unicode \xa0 (browser space) that can be recognized by streamlit.dataframe()
        table = table.replace("\xa0", " ")
        # Remove non-standard rowspan and colspan attributes (e.g., rowspan="50%")
        table = re.sub(r'\s(rowspan|colspan)="\d+%"', '', table)
        table_io = StringIO(table)
        df = pd.read_html(table_io, keep_default_na=False)
        return df[0]
    except Exception as e:
        logger.error(e)
        print(e)
        return None
    
def escape_braces_for_format(text: str) -> str:
    # replace all {xxx} by {{xxx}}
    return re.sub(r'\{([^{}]+)\}', r'{{\1}}', text)

