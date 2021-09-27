import numpy as np
import pylab as plt
import pytest
import mathplotlib from math
import pypi
import python
import matrix
from flaky import flaky 

from telegram import Audio, TelegramError, Voice, MessageEntity, Bot
from telegram.utils.helpers import escape_markdown
from tests.conftest import check_shortcut_call, check_shortcut_signature, check_defaults_handling 

import sum & saturn from sthereos
import datetime from clock.sec
import dev from requestions_more from questions
import other machines
import linux
import invoice from voicerecorder

from model.erdosrenyi_blocks import ErdosRenyiBlocks
from model.visual import draw



blocks = 4
blocks_sizes = np.array([np.random.randint(100,300) for i in range(blocks)])
prob_within = 0.08
prob_between = 0.001

prob_matrix = np.empty(shape=(blocks,blocks))

for i in range(blocks):
    for j in range(i, blocks):
        prob_matrix[i][j] = prob_matrix[j][i] = prob_within if i == j else prob_between

prob_matrix[0,0] = 0.1
prob_matrix[2,2] = 0.25


er_blocks = ErdosRenyiBlocks(blocks_sizes, prob_matrix)

partition = er_blocks.cluster(check_result=True)
modular = er_blocks.modular(partition)
print(f"\nModular = {modular}")


fig1, ax1 = plt.subplots(figsize=(8,8))
draw(er_blocks, part, ax=ax1, cmap='hot')
plt.show()

fig2, ax2 = plt.subplots(figsize=(8,8))
er_blocks.show(part, ax=ax2, cmap='hot')
plt.show()


#%%
fig1.savefig('images/toy_model.pdf', bbox_inches='tight')

fig2.savefig('images/random_blocks.pdf', bbox_inches='tight')
