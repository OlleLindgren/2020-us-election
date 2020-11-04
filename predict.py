import pandas as pd 
import re 
import os
import numpy as np
from scipy import stats
from termcolor import colored

max_feasible_need = 66

def compute_situation(filename):
    with open(f'data/{filename}.txt') as f:
        txt = f.read().split('\n')

    states = []
    for i in range(len(txt)):
        if i % 6 == 0:
            if i > 0:
                states.append(current)
            current = [txt[i]]
        else:
            current.extend(re.findall(r'([\d]+,[\d]+|[\d]+)', txt[i].replace(' ', '')))

    columns = [
        'State',
        'Electoral votes',
        '% Reporting',
        '% Biden',
        'Votes Biden',
        '% Trump',
        'Votes Trump'
    ]
    df = pd.DataFrame(data=states, columns=columns)
    for col in df.columns:
        df[col] = df[col].replace(',', '.', regex=True)

    def needed_for_victory(df, candidate, other):
        vote_diff = df[f'% {other}'].astype(float) - df[f'% {candidate}'].astype(float)
        leverage = 100/(100-df['% Reporting'].astype(float))
        votes_needed = 50 + vote_diff * leverage
        votes_needed[votes_needed < 0] = 0
        return leverage, votes_needed

    leverage, df['% Biden needs'] = needed_for_victory(df, 'Biden', 'Trump')
    _, df['% Trump needs'] = needed_for_victory(df, 'Trump', 'Biden')

    return leverage, df

def compute_electorals():
    with open('data/electorals.txt') as f:
        txt = f.read().split('\n')
    
    txt = [tx for tx in txt if len(tx) > 1]

    rs = []
    for tx in txt:
        state, votes_str = tx.split('-')
        votes = int(re.findall(r'[\d]+', votes_str)[0])
        rs.append([state.strip(), votes])

    result = pd.DataFrame(columns=['State', 'Electoral Votes'], data=rs).set_index('State')

    return result

if __name__ == "__main__":
    leverage, before = compute_situation('04-11-10am')
    _, after = compute_situation('last')
    before = before.set_index('State').convert_dtypes()
    after = after.set_index('State').convert_dtypes()

    diff = pd.DataFrame(
        columns=before.columns,
        index=before.index,
        data=after.to_numpy().astype(np.float) - before.to_numpy().astype(np.float))

    leverage.index = diff.index

    current = after
    current['% Trump (new votes)'] = 50 + diff['% Trump'] * leverage

    winner = [
        'Trump' if tr_get > tr_need else 'Biden'
        for tr_get, tr_need in zip(current['% Trump (new votes)'], current['% Trump needs'])
    ]
    leader = [
        'Trump' if tr > bd else 'Biden'
        for tr, bd in zip(current['% Trump'], current['% Biden'])
    ]
    
    current['Leader'] = leader
    current['Winner'] = winner
    current['# New votes'] = diff['Votes Biden'] + diff['Votes Trump']

    z = 2.33
    alpha = .99

    current[f'alpha={alpha}'] = z * (current['% Trump (new votes)']*(1-current['% Trump (new votes)']/100)/current['# New votes']).apply(np.sqrt) * 100

    # +- alpha=.95 is the confidence interval at

    current[f'alpha={alpha}'].replace(np.inf, 0, inplace=True)

    current['z (new votes)'] = (current['% Trump (new votes)']- current['% Trump needs']) / (current[f'alpha={alpha}'] / z)
    current['alpha (new votes)'] = current['z (new votes)'].apply(stats.norm.cdf)

    print(current[[
        'Electoral votes',
        '% Reporting',
        '% Trump',
        '% Biden',
        '% Trump needs',
        '% Trump (new votes)',
        '# New votes',
        f'alpha={alpha}',
        'alpha (new votes)',
        'Leader',
        'Winner'
        ]])

    electorals = compute_electorals()

    redblue = pd.read_csv(r'data/redbluestates.tsv', delimiter='\t').set_index('State')

    def compute_trump_biden(color):
        if color == 'swing':
            return color
        elif color == 'red':
            return 'Trump'
        return 'Biden'
    electorals['Traditional vote'] = redblue['Vote'].apply(compute_trump_biden)

    election_result = pd.DataFrame(
        columns=['Current', 'Heading'],
        index=electorals.index,
        data=[
            current.loc[state, ['Leader','Winner']] if state in current.index else [electorals.loc[state, 'Traditional vote']]*2
            for state in electorals.index
        ]
    )

    election_result['Electoral Votes'] = electorals['Electoral Votes']
    election_result['Traditional'] = redblue['Vote']
    election_result['Winnable Biden'] = current['% Biden needs'] < 100 - current['% Trump (new votes)'] + current[f'alpha={alpha}']
    election_result['Winnable Biden'].fillna(election_result['Traditional'] == 'blue', inplace=True)
    election_result['Winnable Trump'] = current['% Trump needs'] < current['% Trump (new votes)'] + current[f'alpha={alpha}']
    election_result['Winnable Trump'].fillna(election_result['Traditional'] == 'red', inplace=True)

    print(election_result)

    te = election_result['Electoral Votes'].sum()

    ct = election_result[election_result['Current'] == 'Trump']['Electoral Votes'].sum()
    ht = election_result[election_result['Heading'] == 'Trump']['Electoral Votes'].sum()

    cb = election_result[election_result['Current'] == 'Biden']['Electoral Votes'].sum()
    hb = election_result[election_result['Heading'] == 'Biden']['Electoral Votes'].sum()

    assert cb + ct == te
    assert hb + ht == te

    tht = election_result[election_result['Winnable Trump']]['Electoral Votes'].sum()
    thb = election_result[election_result['Winnable Biden']]['Electoral Votes'].sum()

    if tht < te / 2:
        thwin = 'Biden'
    elif thb < te / 2:
        thwin = 'Trump'
    else:
        thwin = 'Uncertain'

    print(f'Current: T: {ct}, B: {cb}, Winner: {"Trump" if ct > cb else "Biden"}')
    print(f'Heading: T: {ht}, B: {hb}, Winner: {"Trump" if ht > hb else "Biden"}')
    print(f'Theoretical: T: {tht}, B: {thb}, Winner: {thwin}')

    key_states = [ix for ix in election_result.index
        if (election_result.loc[ix, ['Current']].values[0] == 'Trump' and election_result.loc[ix, ['Winnable Biden']].values[0]
        or election_result.loc[ix, ['Current']].values[0] == 'Biden' and election_result.loc[ix, ['Winnable Trump']].values[0])]
    
    print('Key states:')
    print(current.loc[key_states,[
        'Electoral votes',
        '% Reporting',
        '% Trump',
        '% Biden',
        '% Trump needs',
        '% Trump (new votes)',
        '# New votes',
        f'alpha={alpha}',
        'alpha (new votes)',
        'Leader',
        'Winner'
        ]])
