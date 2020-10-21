import pandas as pd
import bert_parser.main as bp

def join_full_in_distinct(full_df, distinct_df):
    # join the aggregated coulms "execution frequency", "median_execution_time" from the trace dataframe into the
    # dataframe with distinct activities
    # group the data frame that contains data from all traces
    full_grouped_ef = full_df.groupby('concept:name')['execution frequency'].mean().reset_index()
    full_grouped_et = full_df.groupby('concept:name')['median_execution_time'].median().reset_index()
    # PREPROCCES the data frame
    # set column header
    full_grouped_et.columns = ['concept:name', 'median_execution_time']
    full_grouped_ef.columns = ['concept:name', 'execution frequency']
    # make key column (concept:name) to lower case to apply join
    full_grouped_ef['concept:name'] = full_grouped_ef['concept:name'].apply(lambda x: x.lower())
    full_grouped_et['concept:name'] = full_grouped_et['concept:name'].apply(lambda x: x.lower())
    # replace the character "_" with " " to apply joining
    full_grouped_ef['concept:name'] = full_grouped_ef['concept:name'].apply(lambda x: x.replace("_", " "))
    full_grouped_et['concept:name'] = full_grouped_et['concept:name'].apply(lambda x: x.replace("_", " "))

    result_df = distinct_df.join(full_grouped_ef.set_index('concept:name'), on='activity')
    result_df = result_df.join(full_grouped_et.set_index('concept:name'), on='activity')

    return result_df

def join_distinct_in_full(distinct_df, full_df):
    full_df['concept:name'] = full_df['concept:name'].apply(lambda x: x.lower())
    full_df['concept:name'] = full_df['concept:name'].apply(lambda x: x.replace("_", " "))

    result_df = full_df.join(distinct_df.set_index('activity'), on='concept:name')
    return result_df

def extract_activity_features(df):
    df_w_actLabels = extract_activity_labels(df)
    df_w_actLabels_ITrelated = extract_IT_relatedness(df_w_actLabels)

    return df_w_actLabels_ITrelated

def extract_activity_features_full_log(df):
    df_full_ef = extract_execution_frequency(df)
    df_full_ef_et = extract_execution_time(df_full_ef)

    return df_full_ef_et


def extract_execution_frequency(df):
    ef_df = df['concept:name'].value_counts().reset_index()
    ef_df.columns = ['concept:name', 'execution frequency']
    result_df = df.join(ef_df.set_index('concept:name'), on='concept:name')
    return result_df

def extract_execution_time(df):
    # print(df[['case:Rfp_id','concept:name', 'time:timestamp']])
    duration = []
    old_trace = ""
    old_time = ""
    for index, row in df.iterrows():
        current_trace = row['case:Rfp_id']
        current_time = row['time:timestamp']

        if current_trace != old_trace:
            # Start a new trace -> first activity has no duration
            duration.append(None)
        else:
            dur = round((current_time-old_time).total_seconds() / 60, 2)
            duration.append(dur)
        old_trace = current_trace
        old_time = current_time
    df['duration_minutes'] = duration
    median_et = df.groupby('concept:name')['duration_minutes'].median().reset_index()
    median_et.columns = ['concept:name', 'median_execution_time']
    result_df = df.join(median_et.set_index('concept:name'), on='concept:name')
    return result_df





def extract_activity_labels(df):
    events = df['concept:name']
    tagged_events = bp.main(events)
    bo = []
    action = []
    actor = []
    actvty = []
    for activity in tagged_events:
        actvty.append(activity)
        i = 0
        bo_indexes = []
        a_indexes = []
        actor_indexes = []

        # retrieve index of BO,A, ACTOR tags
        for tag in tagged_events[activity]:
            if tag == "BO":
                bo_indexes.append(i)
            elif tag == "A":
                a_indexes.append(i)
            elif tag == "ACTOR":
                actor_indexes.append(i)
            i += 1

        activity_list_words = activity.split()
        # iterate through all words of an activity and identify the BO, A, Actor values and create a separate list for
        # each tag
        bo_values = []
        for b in bo_indexes:
            bo_values.append(activity_list_words[b])
        bo.append(" ".join(bo_values))

        a_values = []
        for a in a_indexes:
            a_values.append(activity_list_words[a])
        action.append(" ".join(a_values))

        actor_values = []
        for ac in actor_indexes:
            actor_values.append(activity_list_words[ac])
        actor.append(" ".join(actor_values))

    dict = {"activity": actvty, "business object": bo, "action": action, "executing resource": actor}
    result_df = pd.DataFrame(dict)
    return result_df


def extract_IT_relatedness(df):
    df['IT relatedness'] = pd.Series(dtype='str')
    it_related_terms = ['access', 'Access Control List', 'access time', 'account', 'account name', 'address',
                        'aggregate', 'aggregate data', 'algorithm', 'alias', 'analog', 'Application Layer',
                        'application', 'application program', 'application software', 'archive', 'argument', 'ASCII',
                        'assembler', 'ATM', 'ATM Forum', 'audit', 'authentication', 'authorization',
                        'autonomous system', 'backbone', 'background processing', 'backspace', 'backup', 'bandwidth',
                        'baseband', 'BASIC', 'batch processing', 'batch query', 'binary ', 'binary number', 'bit',
                        'bitmapped terminal', 'BITNET', 'bits per second (bps)', 'block', 'bold', 'booting ', 'break',
                        'bridge', 'broadband', 'broadcast', 'browser', 'buffer', 'bug ', 'bulletin board (BBS)',
                        'BUS topology', 'byte', 'cable', 'carriage return <cr></cr>', 'CD-ROM', 'cell relay', 'channel',
                        'character', 'character set', 'chip', 'client', 'client/server', 'Client-Server Interface',
                        'COBOL', 'code', 'collision', 'column', 'command', 'communications line',
                        'communications program', 'compiler', 'computer', 'concentrator', 'conference', 'configuration',
                        'connect time', 'control character', 'control key', 'copy', 'CPU', 'crash', 'cursor',
                        'cursor control', 'Cyberspace', 'Data Link Layer', 'data', 'data communications', 'data entry',
                        'data processing', 'Dataset', 'database', 'database management system', 'DBMS', 'debug',
                        'default', 'delete key ', 'DHCP', 'dial-up', 'dictionary file', 'digital', 'direct access',
                        'directory', 'disk or diskette ', 'display', 'distributed', 'distributed application',
                        'distributed database', 'distributed file system', 'document', 'documentation', 'DOS',
                        'dot-matrix printer', 'down', 'download', 'downtime', 'drag and drop', 'drive', 'dump', 'edit',
                        'editor', 'e-mail', 'e-mail address', 'e-mail server', 'e-mail service', 'encapsulation',
                        'enter key', 'environment', 'erase', 'error message', 'error checking ', 'Ethernet', 'execute',
                        'fiber optics', 'field', 'file', 'file format', 'file server ', 'folder', 'font', 'foreground',
                        'form', 'form feed', 'format', 'FORTRAN', 'fragment', 'frame', 'freeware', 'frequency', 'FAQ',
                        'FTP', 'FUD', 'function key', 'G', 'garbage', 'gateway', 'GIF', 'gopher', 'graphic',
                        'Groupware', 'GUI', 'handshaking', 'hang', 'hard copy', 'hard disk', 'hardware', 'hardwired',
                        'header', 'help', 'hierarchical file', 'hierarchical file structure', 'host', 'host computer',
                        'HTML', 'HTTP', 'hub', 'hyperlink', 'hypermedia', 'hypertext', 'icons', 'I/O', 'IEEE', 'inbox',
                        'index', 'information hiding', 'information server', 'information superhighway', 'inheritance',
                        'input', 'instance', 'instantiation', 'instruction', 'interactive', 'INTERNET', 'IP ',
                        'IP Address', 'interrupt', 'IRC', 'ISDN', 'ISO', 'job', 'JPEG', 'justify', 'Kermit ', 'key',
                        'keyboard', 'kilobyte(K)', 'LAN', 'LAN e-mail system', 'laserdisc', 'laser printer', 'Layer',
                        'line', 'line editor', 'line printer', 'link', 'LISTSERV', 'load', 'logical record',
                        'login or logon', 'login ID', 'logoff', 'Longitudinal Study', 'LPR', 'lynx', 'machine language',
                        'macro', 'magnetic disk', 'magnetic tape', 'MAIL', 'mailbox', 'MAILER', 'main memory',
                        'mainframe', 'mainframe, minicomputer, micro-computer', 'MB', 'medium', 'memory', 'menu',
                        'message', 'method', 'methodology', 'microcomputer', 'microprocessor', 'Microwave', 'mission',
                        'modem', 'modem setup', 'module', 'monitor', 'Mosaic', 'mouse', 'multimedia', 'multimedia mail',
                        'multiplexer', 'multiuser', 'nesting', 'NetScape', 'Network Layer', 'network', 'nickname',
                        'node', 'noise', 'object', 'object-based', 'object code', 'object-oriented',
                        'object-oriented technology', 'OLE', 'off-line ', 'on-line ', 'Online Service', 'open',
                        'open platform', 'open system', 'OSI ', 'OpenWindows', 'operating system', 'output', 'packet',
                        'parameter', 'parity', 'password', 'peripheral ', 'PC', 'Physical Layer', 'ping', 'pixel',
                        'platform', 'plotter', 'polymorphism', 'port', 'portable', 'post', 'PostScript', 'Power PC',
                        'Presentation layer', 'printer', 'printout', 'procedure', 'process', 'program', 'programmer',
                        'programming', 'prompt', 'protocol', 'public domain', 'quality', 'query', 'queue', 'quit',
                        'RAID', 'RAM', 'random access', 'Re-engineering', 'read', 'read/write', 'realtime', 'record',
                        'record length', 'record type', 'recovery', 'rectangular file', 'reel tape',
                        'relational database', 'relational structure', 'remote', 'remote access', 'resource',
                        'response', 'retiming', 'return key', 'reuse and reuseability', 'reverse engineering', 'ROM',
                        'root directory', 'router', 'routine', 'routing', 'run', 'scanner', 'scheduling', 'screen',
                        'screen editor', 'scroll', 'segment', 'sequential', 'server', 'service (or service provider) ',
                        'session', 'Session Layer', 'shareware', 'shell', 'simulation', 'smiley', 'soft copy',
                        'software', 'software tool', 'sort', 'source code', 'SPARC', 'SPARCstation', 'sponge', 'spool',
                        'spreadsheet', 'SQL', 'storage', 'strategy', 'string', 'striping', 'Sun Microsystems', 'SunOS',
                        'surfing', 'tape density', 'task', 'TCP/IP', 'TEAM', 'telecommunication', 'telecomputing',
                        'TELNET', 'terminal', 'terminal emulation', 'terminal server', 'terabyte', 'text', 'time out',
                        'time series.', 'TN3270', 'toggle', 'token ring', 'topic', 'transfer', 'Transport Layer',
                        'tree', 'UNIX', 'upload', 'URL', 'Usenet', 'user', 'user-friendly', 'userid', 'username ',
                        'utility', 'variable', 'vision', 'virtual', 'virtual terminal', 'VMS', 'virus', 'volume',
                        'wavelength', 'whois', 'window', 'Windows', 'word processor ', 'wordwrap', 'work space',
                        'workstation', 'write', 'WWW', 'X window system', 'X-term']
    for index, row in df.iterrows():
        for term in it_related_terms:
            activity = row["activity"].lower()
            if activity.find(term.lower()) == -1:
                row["IT relatedness"] = False
            else:
                row["IT relatedness"] = True
    return df


