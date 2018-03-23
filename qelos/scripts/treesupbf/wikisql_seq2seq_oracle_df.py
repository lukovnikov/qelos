from qelos.scripts.treesupbf.wikisql_clean import run_seq2seq_oracle_df as runf
import qelos as q

if __name__ == "__main__":
    q.argprun(runf)