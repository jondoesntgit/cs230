import paramiko
import getpass
from pathlib import Path

# TODO: Put these in environment variables
username = input('Username? ')
password = getpass.getpass('Password? ')
assert len(password) > 0

paramiko.util.log_to_file('paramiko.log')

sf = paramiko.Transport(('cardinal.stanford.edu', 22))
sf.connect(username=username, password=password)
sf.auth_interactive_dumb(username)

sftp = paramiko.SFTPClient.from_transport(sf)

# TODO: This is hardcoded. This is usually bad
remote_path = '/afs/.ir/users/j/a/jamwheel/cs230/foo.txt'
local_path = '/tmp/receive.txt'

sftp.get(remote_path, local_path)

# TODO: More hardcoding...
with open('/tmp/send.txt', 'w') as f:
    f.write('Stuff to put on server')
remote_path = '/afs/.ir/users/j/a/jamwheel/cs230/write_test.txt'
local_path = '/tmp/send.txt'
results = sftp.put(local_path, remote_path)
print(results)

