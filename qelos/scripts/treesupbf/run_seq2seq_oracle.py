from qelos.scripts.treesupbf.pasdecode import run_seq2seq_oracle as runf
import qelos as q

if __name__ == "__main__":
    q.argprun(runf)