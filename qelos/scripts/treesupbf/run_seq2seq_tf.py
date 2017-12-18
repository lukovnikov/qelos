from qelos.scripts.treesupbf.pasdecode import run_seq2seq_teacher_forced as runf
import qelos as q

if __name__ == "__main__":
    q.argprun(runf)