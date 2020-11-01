import pandas as pd
import bert_parser.main as bp
import spacy
from pm4py.algo.discovery.footprints import algorithm as footprints_discovery

ACTIVITY_COLUMN_NAME = 'concept:name'
CASE_COLUMN_NAME = 'case:Rfp_id'

def join_full_in_distinct(full_df, distinct_df):
    # join the aggregated coulms "execution frequency", "median_execution_time" from the trace dataframe into the
    # dataframe with distinct activities
    # group the data frame that contains data from all traces
    full_grouped_ef = full_df.groupby('activity')['ef_relative'].mean().reset_index()
    full_grouped_et = full_df.groupby('activity')['median_execution_time'].median().reset_index()
    full_grouped_rel_et = full_df.groupby('activity')['et_relative'].median().reset_index()
    full_grouped_stability = full_df.groupby('activity')['stability'].mean().reset_index()
    # PREPROCCES the data frame
    # set column header
    full_grouped_et.columns = ['activity', 'median_execution_time']
    full_grouped_ef.columns = ['activity', 'ef_relative']
    full_grouped_rel_et.columns = ['activity', 'et_relative']
    full_grouped_stability.columns = ['activity', 'stability']

    result_df = distinct_df.join(full_grouped_ef.set_index('activity'), on='activity')
    result_df = result_df.join(full_grouped_et.set_index('activity'), on='activity')
    result_df = result_df.join(full_grouped_rel_et.set_index('activity'), on='activity')
    result_df = result_df.join(full_grouped_stability.set_index('activity'), on='activity')

    return result_df

# def join_distinct_in_full(distinct_df, full_df):
#     full_df['concept:name'] = full_df['concept:name'].apply(lambda x: x.lower())
#     full_df['concept:name'] = full_df['concept:name'].apply(lambda x: x.replace("_", " "))
#
#     result_df = full_df.join(distinct_df.set_index('activity'), on='concept:name')
#     return result_df


def extract_activity_features(df, xes_log):
    df.rename(columns={ACTIVITY_COLUMN_NAME: "activity"}, inplace=True)
    df['activity'] = df['activity'].apply(lambda x: x.lower())
    df['activity'] = df['activity'].apply(lambda x: x.replace("_", " "))
    df_w_actLabels = extract_activity_labels(df)
    df_w_actLabels_ITrelated = extract_IT_relatedness(df_w_actLabels)
    df_w_actLabels_ITrelated_deterministic = extract_deterministic_standardization_feature(df_w_actLabels_ITrelated, xes_log)
    df_w_actLabels_ITrelated_deterministic_std_fr = extract_failure_rate(df_w_actLabels_ITrelated_deterministic, df)
    df_w_actLabels_ITrelated_Deterministic_std_fr_nr = extract_number_resources(df_w_actLabels_ITrelated_deterministic_std_fr)
    return df_w_actLabels_ITrelated_Deterministic_std_fr_nr


def extract_activity_features_full_log(df):
    df.rename(columns={ACTIVITY_COLUMN_NAME: "activity"}, inplace=True)
    df_full_ef = extract_execution_frequency(df)
    df_full_ef_et = extract_execution_time(df_full_ef)
    df_full_ef_et_stability = extract_stability(df_full_ef_et)
    df_full_ef_et['activity'] = df_full_ef_et['activity'].apply(lambda x: x.lower())
    df_full_ef_et['activity'] = df_full_ef_et['activity'].apply(lambda x: x.replace("_", " "))
    return df_full_ef_et_stability


def extract_number_resources(df):
    action_bo_dict = {}
    for index, row in df.iterrows():
        key = str(row['action']+row['business object'])
        if key in action_bo_dict:
            action_bo_dict[key].append(row['executing resource'])
        else:
            value = [row['executing resource']]
            kv = {key: value}
            action_bo_dict.update(kv)

    for key in action_bo_dict:
        action_bo_dict[key] = list(set(action_bo_dict[key]))

    number_resources_list = []
    for index, row in df.iterrows():
        key = row['action']+row['business object']
        number_resources_list.append(len(action_bo_dict[key]))
    df['number of resources'] = number_resources_list

    return df


def extract_stability(df):
    # Retrieve std deviation
    std_duration = df.groupby('activity')['duration_minutes'].std().reset_index()
    std_duration.columns = ['activity', 'standard_deviation_duration']
    result_df = df.join(std_duration.set_index('activity'), on='activity')
    list_stability = []
    # for each activity compute the stability by using a normalized variance
    for index, row in result_df.iterrows():
        list_stability.append(row['standard_deviation_duration'] ** 2 / (row['ef'] * row['avg_execution_time']))
    result_df['stability'] = list_stability
    result_df['stability'].fillna(0, inplace=True)
    return result_df


def extract_failure_rate(df, full_df):
    df_act_trace_occurrence = full_df.groupby('activity')[CASE_COLUMN_NAME].nunique().reset_index()
    df_act_trace_occurrence.columns = ['activity', 'trace_occurrence']
    df_act_count = full_df['activity'].value_counts().reset_index()
    df_act_count.columns = ['activity', 'activity_count']
    df_actcount_traceocc = df_act_trace_occurrence.join(df_act_count.set_index('activity'), on='activity')

    avg_exec = []
    for index, row in df_actcount_traceocc.iterrows():
         avg_exec.append(row['activity_count'] / row['trace_occurrence'])
    df_actcount_traceocc['failure_rate'] = avg_exec
    df_actcount_traceocc.drop(columns=['activity_count', 'trace_occurrence'], inplace=True)
    df = df.join(df_actcount_traceocc.set_index('activity'), on='activity')

    return df


def extract_deterministic_standardization_feature(df, log):
    fp_log=footprints_discovery.apply(log, variant=footprints_discovery.Variants.ENTIRE_EVENT_LOG)
    # Footprint DF from PM4PY
    directly_follows = fp_log['sequence']
    # dict for following activity
    df_dict = {}
    # dict for preceding activity
    dp_dict = {}
    # Create dicts with activity as key and list of preceding / following activities as value
    for val in directly_follows:
        tuple_list = list(val)
        tuple_list[0] = tuple_list[0].lower().replace('_', ' ')
        tuple_list[1] = tuple_list[1].lower().replace('_', ' ')
        if tuple_list[0] not in df_dict:
            value = [tuple_list[1]]
            kv = {tuple_list[0]: value}
            df_dict.update(kv)
        else:
            value = df_dict.get(tuple_list[0])
            value.append(tuple_list[1])
            kv = {tuple_list[0]: value}
            df_dict.update(kv)
        if tuple_list[1] not in dp_dict:
            dp_value = [tuple_list[0]]
            dp_kv = {tuple_list[1]: dp_value}
            dp_dict.update(dp_kv)
        else:
            dp_value = dp_dict.get(tuple_list[1])
            dp_value.append(tuple_list[0])
            dp_kv = {tuple_list[1]: dp_value}
            dp_dict.update(dp_kv)
    # List for determinism following
    df_list = []
    # List for standardization following
    sf_list = []
    for key in df_dict:
        df_list.append([key, True if len(df_dict[key]) == 1 else False ])
        sf_list.append([key, len(df_dict[key])])
    df_dataframe = pd.DataFrame(df_list, columns=['activity', 'deterministic_following_activity'])
    sf_dataframe = pd.DataFrame(sf_list, columns=['activity', 'following_activities_standardization'])
    # List for determinism preceding
    dp_list = []
    # List for standardization preceding
    sp_list = []
    for key in dp_dict:
        dp_list.append([key, True if len(dp_dict[key]) == 1 else False ])
        sp_list.append([key, len(dp_dict[key])])
    dp_dataframe = pd.DataFrame(dp_list, columns=['activity', 'deterministic_preceding_activity'])
    sp_dataframe = pd.DataFrame(sp_list, columns=['activity', 'preceding_activities_standardization'])
    # Join Dataframes
    result_df = df.join(df_dataframe.set_index('activity'), on='activity')
    result_df = result_df.join(dp_dataframe.set_index('activity'), on='activity')
    result_df = result_df.join(sf_dataframe.set_index('activity'), on='activity')
    result_df = result_df.join(sp_dataframe.set_index('activity'), on='activity')
    result_df['deterministic_following_activity'].fillna(False, inplace=True)
    result_df['deterministic_preceding_activity'].fillna(False, inplace=True)
    result_df['following_activities_standardization'].fillna(0, inplace=True)
    result_df['preceding_activities_standardization'].fillna(0, inplace=True)
    return result_df


def extract_execution_frequency(df):
    relative_ef_df = df['activity'].value_counts(normalize=True).reset_index()
    ef_df = df['activity'].value_counts().reset_index()
    relative_ef_df.columns = ['activity', 'ef_relative']
    ef_df.columns = ['activity', 'ef']
    result_df = df.join(relative_ef_df.set_index('activity'), on='activity')
    result_df = result_df.join(ef_df.set_index('activity'), on='activity')
    return result_df

def extract_execution_time(df):
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
    #median et
    median_et = df.groupby('activity')['duration_minutes'].median().reset_index()
    median_et.columns = ['activity', 'median_execution_time']
    result_df = df.join(median_et.set_index('activity'), on='activity')
    #avg et
    avg_et = df.groupby('activity')['duration_minutes'].mean().reset_index()
    avg_et.columns = ['activity', 'avg_execution_time']
    result_df = result_df.join(avg_et.set_index('activity'), on='activity')
    #relative et
    grouped_sum_et = df.groupby('activity')['duration_minutes'].sum().reset_index()
    sum_et = df['duration_minutes'].sum()
    grouped_sum_et.columns = ['activity', 'sum_execution_time']
    result_df = result_df.join(grouped_sum_et.set_index('activity'), on='activity')
    relative_durations = []
    for index, row in result_df.iterrows():
        relative_durations.append(row['sum_execution_time'] / sum_et)
    result_df['et_relative'] = relative_durations
    return result_df

def extract_activity_labels(df):
    events = df['activity']
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
        if bo_indexes:
            for b in bo_indexes:
                bo_values.append(activity_list_words[b])
        else:
            bo_values.append('missing')
        bo.append(" ".join(bo_values))

        a_values = []
        if a_indexes:
            for a in a_indexes:
                a_values.append(activity_list_words[a])
        else:
            a_values.append('missing')
        action.append(" ".join(a_values))

        actor_values = []
        if actor_indexes:
            for ac in actor_indexes:
                actor_values.append(activity_list_words[ac])
        else:
            actor_values.append('missing')
        actor.append(" ".join(actor_values))

    dict = {"activity": actvty, "business object": bo, "action": action, "executing resource": actor}
    result_df = pd.DataFrame(dict)
    # Utilize attribute org:resource if exists
    result_df = add_alternative_resource_attributes(df, result_df)
    return result_df


def add_alternative_resource_attributes(df, result_df):
    temp_df = pd.merge(df, result_df[["activity", "executing resource"]], on="activity", how="left")
    if 'org:resource' in temp_df:
        gdf = temp_df.groupby('activity')['org:resource'].apply(lambda x: ','.join(x)).reset_index()
        gdf['org:resource'] = gdf['org:resource'].apply(lambda x: ', '.join(sorted(set(x.split(',')))))
        result_df = result_df.join(gdf.set_index('activity'), on='activity')

    # Utilize attribute org:role if exists
    if 'org:resource' in temp_df:
        gdf = temp_df.groupby('activity')['org:role'].apply(lambda x: ','.join(x)).reset_index()
        gdf['org:role'] = gdf['org:role'].apply(lambda x: ', '.join(sorted(set(x.split(',')))))
        gdf['org:role'] = gdf['org:role'].apply(lambda x: x.lower())
        gdf['org:role'] = gdf['org:role'].apply(lambda x: x.replace("_", " "))
        result_df = result_df.join(gdf.set_index('activity'), on='activity')
    return result_df


def extract_IT_relatedness(df):
    nlp = spacy.load("en_core_web_md")
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
    it_related_terms = ' '.join(it_related_terms)
    it_tokens = nlp(it_related_terms)
    max_similarities = []
    for index, row in df.iterrows():
        similarity_dict = {}
        activity = row["activity"].lower()
        activity_tokens = nlp(activity)
        for activity_token in activity_tokens:
            for it_token in it_tokens:
                if activity_token.has_vector and it_token.has_vector:
                    similarity_dict[activity_token.text + it_token.text] = activity_token.similarity(it_token)
        max_similarities.append(max(similarity_dict.values()))

    df['IT relatedness'] = max_similarities
    return df


