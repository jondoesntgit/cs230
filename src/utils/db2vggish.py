#!/usr/bin/env python

"""
Fetches a worklist from a databse and converts youtube slugs to vggish features
and stores them on AWS
"""

import psycopg2
import os
import dotenv
dotenv.load_dotenv()

PSQL_HOSTNAME = os.getenv('PSQL_HOSTNAME')
PSQL_USERNAME = os.getenv('PSQL_USERNAME')
PSQL_PASSWORD = os.getenv('PSQL_PASSWORD')
PSQL_DATABASE = os.getenv('PSQL_DATABASE')


def main(conn):
    print(conn)


if __name__ == '__main__':
    with psycopg2.connect(
            host=PSQL_HOSTNAME,
            database=PSQL_DATABASE,
            user=PSQL_USERNAME,
            password=PSQL_PASSWORD) as conn:

        main(conn)
