import logfire
import os
from dotenv import load_dotenv

load_dotenv()
logfire.configure(token=os.getenv("write_token"))

logfire.info('check api validation')
print('sadf')