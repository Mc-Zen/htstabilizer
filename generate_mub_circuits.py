import qiskit
from src.htstabilizer.mub_circuits import get_mub_circuits, get_mubs
from src.htstabilizer.stabilizer_circuits import *


def parse_mub(spec: str):
    mub = []
    sets = [s.strip("{}") for s in spec.split("},{")]
    # n = len(sets[0].split(",")[0])
    for set in sets:
        mub.append(Stabilizer(set.split(",")))
    return mub


def to_circ_spec(circuit: QuantumCircuit) -> str:
    string = ""
    for instruction in circuit:
        op = instruction.operation

        string += op.name + ",".join([str(circuit.find_bit(qubit)[0]) for qubit in instruction.qubits])
        assert len(op.params) == 0
        string += " "
    return string

def optimize6qubitMubCircuits():
    circuits = get_mub_circuits(6, "all")
    mubs = get_mubs(6, "all")

    lines = []
    for set, circuit in zip(mubs, circuits):
        circuit = qiskit.transpile(circuit, basis_gates=["h", "s", "sdg", "cx", "cz"], optimization_level=3)

        ops = str(set)
        ops = ops[ops.find("[") + 1:ops.find("]")].replace('"', "")
        line = f"{ops}:{to_circ_spec(circuit)}"
        lines.append(line)


    with open(f"src/htstabilizer/data/mub6-all.txt", "w") as file:
        file.write("\n".join(lines))
optimize6qubitMubCircuits()
def write_mub_circuit_file(spec: str, connectivity: str):
    mub = parse_mub(spec)
    num_qubits = mub[0].num_qubits
    circuits = []
    for set in mub:
        circuit = get_readout_circuit(set, connectivity)
        circuit = qiskit.transpile(circuit, basis_gates=["h", "s", "sdg", "cx", "cz"], optimization_level=3)
        circuits.append(circuit)
        print(circuit)

    lines = []
    total_cost = 0
    max_cost = 0
    max_depth = 0
    for set, circuit in zip(mub, circuits):
        cx_circuit = qiskit.transpile(circuit, basis_gates=["cx", "h", "s"])
        cost = cx_circuit.count_ops().get("cx", 0)
        max_cost = max(max_cost, cost)
        max_depth = max(max_depth, cx_circuit.depth(lambda instr: instr.operation.name == "cx"))
        total_cost += cost

        ops = str(set)
        ops = ops[ops.find("[") + 1:ops.find("]")].replace('"', "")
        line = f"{ops}:{to_circ_spec(circuit)}"
        lines.append(line)

    info_line = f"{total_cost}:{max_cost}:{max_depth}"
    lines.insert(0, info_line)
    print(lines)

    with open(f"src/htstabilizer/data/mub{num_qubits}-{connectivity}.txt", "w") as file:
        file.write("\n".join(lines))

# write_mub_circuit_file("{{IZYII,IIIIZ,IIIZI,YIYII,IZIIZ},{XZZII,ZYZII,ZZYII,IIIXI,IIIIX},{XIXII,ZYZIZ,ZZYZI,YIYXI,IZIIY},{XZZIZ,ZYZZI,XZIII,IZIXZ,IZYZX},{XZZZI,XYXII,ZIYIZ,IZYYI,YIYIY},{ZZXII,ZXZIZ,ZIIZI,YIYXZ,IZIZY},{XIZIZ,ZXXZI,XZIIZ,IZIYZ,YZIZX},{XIXZI,XYXIZ,ZIYZZ,YZIYI,YZYIX},{ZZXIZ,ZXZZZ,XIYZI,YZYXI,IIYIY},{XIZZZ,XXZZI,XIIII,IIYXZ,IZYZY},{ZIZZI,XXXII,ZZIIZ,IZYYZ,YIYZY},{ZIXII,ZYXIZ,ZIIZZ,YIYYZ,YZYZY},{XZXIZ,ZXXZZ,XZIZZ,YZYYZ,YIIZY},{XIXZZ,XYXZZ,XIIZZ,YIIYZ,YIIZX},{ZZXZZ,XXXZZ,XZYZZ,YIIYI,YIIIX},{ZIXZZ,XYZZZ,XZYZI,YIIXI,IIYIX},{ZZZZZ,XYZZI,XZYII,IIYXI,IZYIY},{ZZZZI,XYZII,ZZIII,IZYXZ,IIIZY},{ZZZII,ZYXII,ZIIIZ,IIIYZ,YIYZX},{XZXII,ZXXIZ,ZZYZZ,YIYYI,YZYIY},{XIXIZ,ZYZZZ,XZIZI,YZYXZ,IIYZY},{XZZZZ,XYXZI,XIIIZ,IIYYZ,YZIZY},{ZZXZI,XXXIZ,ZZIZZ,YZIYZ,YZYZX},{ZIXIZ,ZYXZZ,XIYZZ,YZYYI,YIIIY},{XZXZZ,XXZZZ,XIIZI,YIIXZ,IIYZX},{ZIZZZ,XXXZI,XZYIZ,IIYYI,YZIIY},{ZIXZI,XYZIZ,ZZIZI,YZIXZ,IZIZX},{ZZZIZ,ZYXZI,XIYIZ,IZIYI,YZIIX},{XZXZI,XXZIZ,ZIYZI,YZIXI,IZIIX},{ZIZIZ,ZXZZI,XIYII,IZIXI,IZYIX},{XIZZI,XXZII,ZIYII,IZYXI,IIIIY},{ZIZII,ZXZII,ZIIII,IIIXZ,IIIZX},{XIZII,ZXXII,ZZYIZ,IIIYI,YIYIX}}", "linear")
# assert False 


# write_mub_circuit_file("{{XX,YY},{XZ,YX},{ZX,ZI},{XY,XI},{YZ,YI}}", "all")

write_mub_circuit_file("{{XZX,YZY,IZI},{ZZX,YYX,IIX},{YII,IXZ,IZX},{XIZ,YXX,ZIY},{ZIX,XYY,YIZ},{IZY,IYZ,ZZY},{XZZ,XXY,XII},{IIY,ZYI,XZI},{YZI,ZXI,YZZ}}", "all")
# write_mub_circuit_file("{{ZZI,IIZ,IZI},{XII,XYI,IIX},{YZI,XYZ,IZX},{XIZ,XXI,ZZY},{XZI,YXZ,IZY},{YZZ,XXZ,ZIY},{XZZ,YYZ,ZIX},{YIZ,YYI,ZZX},{YII,YXI,IIY}}", "linear")

# write_mub_circuit_file("{{XZYY,YZYX,ZZXI,IZII},{ZIIY,ZYYI,ZZZI,YZYZ},{YZYI,XXIX,IIYI,YIYZ},{XZYZ,IXZI,ZIZI,XZYI},{IZXY,ZXYI,IZZZ,ZZXY},{ZZIY,IYYZ,YZIX,XZZZ},{IIIX,YYXX,IZYI,XIYI},{YIZZ,IYZI,IIZZ,IZXX},{IIXY,IXYZ,XZIY,IZIY},{IZIX,XYXY,XZXX,YIZI},{XIZI,XYIX,ZIYZ,ZIXY},{XIYZ,ZXZZ,YIIX,YZZI},{ZZXX,YXXX,ZZYZ,IIXX},{YZZZ,ZYZZ,XIIY,ZZIX},{ZIXX,XXXY,YZXY,ZIIX},{XZZI,YYIY,YIXY,IIIY},{YIYI,YXIY,XIXX,XIZZ}}", "all")
# write_mub_circuit_file("{{ZZII,IIIZ,IIZI,IZII},{XIII,XYII,IIXI,IIIX},{YZII,XYIZ,IIYI,IZIX},{XIIZ,XYZI,IZXI,ZZIY},{XIZI,XXII,ZZXZ,IIZY},{XZII,YXIZ,IIYZ,IZZX},{YZIZ,XYZZ,IZYI,ZIIY},{XIZZ,XXZI,ZIXZ,ZZZX},{XZZI,YYIZ,ZZYI,IZIY},{YIIZ,YXZI,IZXZ,ZZZY},{YZZI,XXIZ,ZZYZ,IZZY},{XZIZ,YXZZ,IZYZ,ZIZY},{YZZZ,XXZZ,ZIYZ,ZIZX},{XZZZ,YYZZ,ZIYI,ZIIX},{YIZZ,YYZI,ZIXI,ZZIX},{YIZI,YYII,ZZXI,IIIY},{YIII,YXII,IIXZ,IIZX}}", "linear")
# write_mub_circuit_file("{{ZZII,IIIZ,IIZI,IZII},{XIII,XYII,IIXI,IIIX},{YZII,XYIZ,IIYI,IZIX},{XIIZ,XYZI,IZXI,ZZIY},{XIZI,XXII,ZZXZ,IIZY},{XZII,YXIZ,IIYZ,IZZX},{YZIZ,XYZZ,IZYI,ZIIY},{XIZZ,XXZI,ZIXZ,ZZZX},{XZZI,YYIZ,ZZYI,IZIY},{YIIZ,YXZI,IZXZ,ZZZY},{YZZI,XXIZ,ZZYZ,IZZY},{XZIZ,YXZZ,IZYZ,ZIZY},{YZZZ,XXZZ,ZIYZ,ZIZX},{XZZZ,YYZZ,ZIYI,ZIIX},{YIZZ,YYZI,ZIXI,ZZIX},{YIZI,YYII,ZZXI,IIIY},{YIII,YXII,IIXZ,IIZX}}", "star")
# write_mub_circuit_file("{{XXYZ,IIIZ,YXXZ,YYYZ},{ZXYZ,IXII,YXZZ,YXYY},{YIII,IXIZ,IIYI,IZIX},{ZXYI,YIXZ,IZXI,ZIIY},{XIZI,YZYZ,ZIXZ,IIZY},{XZII,XIYI,IIYZ,YYXY},{YIIZ,YIXI,YYIZ,XYYX},{XIZZ,IYZI,XYZI,XXXY},{ZYXZ,ZYIZ,XXIZ,IZIY},{IYYI,ZXZI,IZXZ,XXXX},{IXXZ,YZYI,XXII,YYXX},{XZIZ,ZXZZ,YYII,ZZZY},{IXXI,IYZZ,ZZYZ,ZZZX},{ZYXI,XZXI,ZZYI,XYYY},{YZZZ,XZXZ,XYZZ,ZIIX},{YZZI,ZYII,ZIXI,YXYX},{IYYZ,XIYZ,YXZI,IIZX}}", "cycle")

# write_mub_circuit_file("{{XZYIY,YZYIX,ZXXXZ,XYIYX,XXZYY},{XXXYZ,YYYIY,YZZIY,ZXXZZ,XYZYI},{IYZYX,IXIIZ,XYYXX,YZXXY,IZIIY},{ZYZYY,XZZXX,ZXZYZ,YIYXX,ZIIZX},{YIIZI,ZIYYZ,ZYIYI,XZYYY,YZXII},{IZXIY,ZZXYI,IXIXZ,IIZXZ,ZYXXX},{IIYIX,IIXXZ,XYXYY,XXZIY,YYIXI},{ZZYZY,XZIYY,IZXZZ,IXYIZ,ZYYYY},{YIZIZ,IYIZZ,XZIZY,XXIZX,XIYII},{ZXZXY,XYXZY,IZYII,ZZIXZ,XZYZI},{YXYXZ,IYZII,YXYYY,ZIIYZ,XYIXZ},{ZXIYX,YIZYY,YYYXY,ZXYII,YIXZI},{XZIIZ,YZZXY,YZIZX,IZZYZ,ZZZZY},{XIIZZ,YYXZX,XXXXY,XIXYX,XIXZZ},{XXYXI,XIIXY,IYZXI,ZZZYI,YXIYI},{YZZZZ,IZYXI,YXXXX,IYYZZ,ZZIIX},{ZIXZX,YIIXX,XIIIY,XIYXY,YYZYZ},{XZZZI,XXXIY,IYIYZ,IXXZI,IIIZY},{YYYYZ,IZXYZ,XZZIX,YZYYX,ZXYXY},{ZIYIY,XYYIX,ZXIXI,XYIIX,XXIYZ},{YXXYI,ZIXXI,IIYZI,ZYYZI,YXZXZ},{IZYZX,IXZZI,YIIIX,IYXII,IXYXX},{ZYIXX,YXXIX,XIZZX,YYIIY,IZZZX},{XYYYI,XXYZX,ZIYZZ,YIXYY,IYYYX},{YYXXI,ZXZZZ,ZYZXZ,YXIZY,IYXXY},{IYIXY,ZZYXZ,ZZYIZ,YXZIX,ZIZIY},{IIXZY,ZYZIZ,ZZXZI,XZXXX,XXZXI},{IXIYY,ZYIZI,IXZYI,ZYXIZ,XZXIZ},{IXZXX,IIYYI,YIZZY,ZIZXI,YIYIZ},{ZZXIX,YXYZY,YYXYX,IZIXI,IXXYY},{XYXXZ,YZIYX,XXYYX,YYZZX,ZXXYX},{XIZII,XIZYX,ZIXII,XYZZY,YZYZZ},{YZIII,ZXIII,IIXIZ,IIIYI,IIZIX}}", "all")
# write_mub_circuit_file("{{ZZYYZ,ZXIZI,YXYXX,YIXYI,XXZYY},{XYXZX,ZYYYY,IIXZI,IIXIX,YXIII},{YXZXY,IZYXY,YXZYX,YIIYX,ZIZYY},{YZXIX,XZIZZ,YIIXI,XXYYZ,ZZIZY},{ZZZYI,XYZIY,XXYXY,XYXZZ,ZIXXI},{ZYIXX,YZXII,XYXIY,XXIXX,XXXZZ},{IZYXZ,YIYXI,XXIYI,ZIIZY,XZXXY},{IIXIZ,YZZZY,ZIIIZ,ZYIXZ,YXYZY},{IZIYX,IYZXX,ZYIYY,IIZZZ,IZXIX},{YYIIY,IIZZI,IIZIY,YYIII,IYIIY},{YIIYZ,ZYIXI,YYIZX,YZXIZ,XXZIX},{XYZIZ,XIZYZ,YZXZY,ZIYII,ZXIXZ},{ZIIZI,XXYYI,ZIYZX,XIXXY,IYYZI},{ZXXZZ,IYXYZ,XIXYZ,YZZZX,ZZYII},{YYYZI,YYYZX,YZZII,XYZIX,XYIXX},{IYXYY,XXIXY,XYZZI,ZZXXI,ZYXZX},{ZXZIX,YIIYY,ZZXYX,XZIZI,YZYXZ},{IIZZX,IXYZZ,XZIIX,IYZXY,ZXYYX},{YXXYI,YXZXZ,IYZYZ,XIZYI,YIZZX},{IXIII,ZIIZX,XIZXX,IXYZI,XIYIY},{XIZYY,YYIIZ,IXYIX,ZXZIZ,YYZXI},{IYZXI,ZZXXZ,ZXZZY,IZYXX,YYXYY},{XZYII,IZIYI,IZYYI,IZIYZ,IIIYZ},{YZZZZ,ZXXZY,IZIXY,YXXYY,YZIYX},{XXYYX,ZXZII,YXXXZ,IYXYI,XZZYI},{XXIXZ,XZYIX,IYXXX,ZYYYX,IZZZZ},{ZZXXY,ZIYIZ,ZYYXI,YYYZY,YIXIZ},{XIXXI,IIXIY,YYYIZ,IXIIY,ZYZIZ},{YIYXX,XIXXX,IXIZZ,XZYIY,IXZXY},{ZIYIY,ZZZYX,XZYZZ,YIYXZ,XYYYZ},{XZIZY,YXXYX,YIYYY,ZZZYY,XIIZI},{IXYZY,XYXZI,ZZZXZ,ZXXZX,IIYXX},{ZYYYZ,IXIIX,ZXXII,YXZXI,IXXYI}}", "linear")
# write_mub_circuit_file("{{YZXYZ,IIIIZ,YXYIZ,XZYYZ,ZXZXZ},{IXYZZ,YIYZZ,ZYYXI,XZXZZ,XZXYY},{YYZXI,YIYZI,XZIXZ,IIZXI,YYYZX},{IXYZI,IXIZI,YXIZZ,YYYYI,XXYIY},{YIIZI,ZZIXI,IZXIZ,XXYXZ,IIZIY},{XYIXI,XXXYI,ZIXZI,IIZXZ,IZIZY},{ZIXYI,YYXXZ,YXIZI,IZIYZ,IYIYX},{IZXXZ,ZZIXZ,YYZII,IYIZI,ZXIXX},{XYIXZ,ZIZYZ,YZZXZ,ZXIII,IXZXY},{XXZYZ,ZXZZI,XIZYZ,IXZIZ,XXYIX},{XIZZI,IYZII,YIIYI,XXYXI,YXXIX},{YZZII,ZYIIZ,ZIXZZ,YXXXI,XIYXX},{XZIIZ,YYXXI,IIYZZ,XIYII,ZZZZY},{IZXXI,XYYXI,ZXXYZ,ZZZYZ,ZZZZX},{ZZYXI,YZXII,XYIII,ZZZYI,XYXZY},{IYXII,IIIYZ,XYIIZ,XYXYZ,IXZXX},{YXIYZ,IIIYI,ZZYII,IXZII,ZIIIY},{YXIYI,YXYYZ,YIIYZ,ZIIXZ,ZYZYY},{IIYYZ,ZYIII,XXZZI,ZYZZZ,YXXIY},{XZIII,IZZXZ,XZIXI,YXXXZ,ZXIXY},{YYZXZ,IXIZZ,IIYZI,ZXIIZ,YIXXX},{YIIZZ,XYYXZ,XIZYI,YIXII,IYIYY},{ZZYXZ,IYZIZ,IXYYZ,IYIZZ,XIYXY},{YZZIZ,XZYII,YZZXI,XIYIZ,XYXZX},{ZYYII,ZXZZZ,ZXXYI,XYXYI,YIXXY},{XIZZZ,YZXIZ,ZZYIZ,YIXIZ,YZYYX},{IYXIZ,YXYYI,IXYYI,YZYZI,IZIZX},{IIYYI,XZYIZ,IYXXZ,IZIYI,YZYYY},{ZYYIZ,XIXZI,YYZIZ,YZYZZ,YYYZY},{ZXXZI,ZIZYI,IYXXI,YYYYZ,ZIIIX},{XXZYI,XIXZZ,IZXII,ZIIXI,XZXYX},{ZXXZZ,XXXYZ,XXZZZ,XZXZI,ZYZYX},{ZIXYZ,IZZXI,ZYYXZ,ZYZZI,IIZIX}}", "star")
# write_mub_circuit_file("{{IXIYI,ZZIZX,XXIYY,IIZII,YXIYZ},{ZZIII,IZIXY,IIXII,YYIZI,ZZIZZ},{ZYIYI,ZIIYZ,XXXYY,YYZZI,XYIXI},{IIIZX,XYIZI,IIYII,IZIXZ,YZIZX},{YYIYY,IZZXY,YXXYZ,ZYIZY,IIZIY},{ZZZII,YYIZX,XIXIY,XXZIX,IZIZY},{XYIYZ,XZIXI,ZZYZX,XYIZX,YZZZX},{YZIIY,ZIZYZ,ZIXIX,ZYZZY,YXZYX},{IIZZX,ZZIXZ,XIYIY,ZIZYY,XZIZI},{IZIIX,XZZXI,XYYXY,IYIZZ,XIIII},{YZZIY,XXZII,YIXIZ,IXIIZ,XXZYI},{YXZXY,YZIXX,YZXZZ,IIZYZ,IZZZY},{XZIIZ,YIIYX,YYYXZ,XYZZX,IYZXY},{XIIZZ,YXZIX,ZIYIX,XZZXX,ZXZYZ},{XXZXZ,ZZZXZ,ZXYYX,YIZYI,YIZIX},{IZZIX,ZYZZZ,IYYXI,ZXZIY,YIIIX},{IYZYX,IXZIY,XZYZY,ZXIIY,IXIYY},{ZXZXI,XIZYI,XZXZY,XIIYX,XXIYI},{YIZZY,XIIYI,ZYXXX,IIIYZ,YYZXX},{YIIZY,ZXIIZ,YYXXZ,ZZZXY,XYZXI},{IXIXX,YXIIX,XXYYY,IZZXZ,IYIXY},{XXIXZ,XYZZI,YXYYZ,XZIXX,XIZII},{YYZYY,YYZZX,ZXXYX,IXZIZ,ZIZIZ},{XYZYZ,ZYIZZ,YZYZZ,YXZII,XZZZI},{IYIYX,YIZYX,IZYZI,IYZZZ,ZXIYZ},{XIZZZ,IIZYY,YIYIZ,YIIYI,IXZYY},{ZIZZI,YZZXX,IYXXI,XIZYX,ZIIIZ},{XZZIZ,IXIIY,ZYYXX,YXIII,ZYZXZ},{ZXIXI,ZXZIZ,IZXZI,YZZXI,YXIYX},{IXZXX,IIIYY,IXYYI,ZIIYY,ZYIXZ},{ZIIZI,IYZZY,XYXXY,YZIXI,IIIIY},{ZYZYI,XXIII,IXXYI,XXIIX,YYIXX},{YXIXY,IYIZY,ZZXZX,ZZIXY,ZZZZZ}}", "cycle")
# write_mub_circuit_file("{{ZZXXX,YYIZY,ZYXZX,YYXXX,XXZYX},{IZIIZ,YYXZY,IZXZZ,YZXIZ,ZYZXX},{ZIXXY,IIXII,ZXIIY,IXIXY,YZIZI},{YXIZX,XIIIZ,YXIYY,ZYYYY,ZZZZX},{ZXXZY,IIIYZ,XYYXY,YYXYZ,ZYYZY},{YXXXY,ZZYXZ,IYXXZ,YZIYI,XXXIX},{XYZYY,YZXXY,IZIXI,IIZXZ,XXYYI},{IYIYZ,YYIXX,YIZYZ,IIIZY,YZXXZ},{IZXYI,IXZYY,YIIIY,ZYZIX,XIXYX},{YIYXZ,IXIIZ,XYZZX,IXZZZ,XXZIZ},{YIXZY,ZZZZI,YXZIZ,IIXXX,IIZIZ},{XYYIX,IIZIY,YIXYX,XXXXX,ZZIXI},{YXYZZ,IXXYI,ZXXYX,YYYIY,IIXIX},{YIIXX,XIXYI,IYYZY,XXZXZ,YYXZZ},{ZXIXX,YZYZZ,ZXZYZ,ZZZYX,XIIIY},{IYZIY,XIZYY,XZZXX,IXYXI,ZYXXZ},{ZXYXZ,ZYZXI,YXYYI,YZZIX,YZYZY},{XZYYX,IIYYX,IZZZX,ZYIYI,YYYXY},{YXZXI,YYZZI,XYIXI,ZZIII,YYZZX},{IZYIX,ZZIXX,XZIZI,ZZXYZ,IXXYX},{XYXYI,ZYIZX,XZXXZ,XIZZZ,YYIXI},{XZXII,ZYXXY,ZIZIZ,ZZYIY,ZZYXY},{XZIYZ,XXZIY,XZYZY,YYIII,XXIYY},{ZIYZZ,ZYYZZ,IYIZI,IIYZI,ZYIZI},{XZZIY,YZIZX,YIYII,YZYYY,IIIYY},{IYXII,IXYIX,IZYXY,XXYZI,IXZYZ},{YIZZI,YYYXZ,ZXYII,XIXZX,ZZXZZ},{IZZYY,XIYIX,ZIXIX,YYZYX,YZZXX},{ZXZZI,XXXII,IYZXX,ZYXIZ,IXIIY},{ZIIZX,YZZXI,XYXZZ,XIYXI,IXYII},{IYYYX,ZZXZY,ZIYYI,XIIXY,XIZYZ},{XYIIZ,XXYYX,ZIIYY,IXXZX,IIYYI},{ZIZXI,XXIYZ,YXXIX,XXIZY,XIYII}}", "T")
# write_mub_circuit_file("{{YXYZZ,IIIIZ,YYYXI,XYYZZ,XXXZI},{XZIXZ,IZXII,ZYZXZ,XXYYZ,ZZIXY},{ZYYYI,IZXIZ,XIXIZ,IZIXI,YYXYY},{XZIXI,YXZXI,YIXYI,IIZXZ,ZIIZX},{ZXYIZ,XXZZZ,YZYYZ,XYYII,YXYYY},{IXYYI,XYIZI,ZXZZI,IZIXZ,IIZZY},{IYXYZ,IIXYZ,YIXYZ,YYXIZ,YYYIY},{XIIZI,XXZZI,IXIZZ,IIIZZ,ZIZXY},{IXYYZ,ZIYYI,YZXIZ,XYXYZ,IZZXX},{YIZZZ,XYZXI,ZXIXZ,YXXYI,ZIIZY},{IYYIZ,IIYII,IYIXI,XYYIZ,IZIZY},{XIZXZ,ZZYIZ,ZXZZZ,YXYIZ,XYXIX},{YZZXI,IIXYI,IYZZZ,ZIZZI,ZZZZY},{XIIZZ,ZZXYI,XZYII,XXXIZ,ZZZZX},{YZIZZ,YYIXZ,ZYIZZ,XXXII,XXXYX},{ZYXII,IZYYI,ZYIZI,ZZZXI,IZZXY},{XZZZZ,IZYYZ,XIYYI,YXXYZ,XYYYY},{XZZZI,YXIZZ,IYIXZ,ZIIXZ,XXYIX},{ZXXYI,ZZYII,XZXYZ,ZZIZI,IZIZX},{YZZXZ,YYZZI,XIXII,YXYII,ZIZXX},{ZYYYZ,YXZXZ,IYZZI,XYXYI,YXXIX},{ZXYII,ZZXYZ,ZXIXI,IZZZI,YYYIX},{YZIZI,IIYIZ,YIYII,IIIZI,XYXIY},{XIZXI,XXIXZ,YZXII,ZIZZZ,XXXYY},{IXXII,XYZXZ,XZYIZ,ZZZXZ,YXXIY},{IYYII,YYIXI,XIYYZ,IZZZZ,IIIXX},{ZYXIZ,YXIZI,YIYIZ,YYYYI,IIZZX},{ZXXYZ,XXIXI,IXZXI,YYXII,IIIXY},{IXXIZ,ZIXIZ,IXIZI,YYYYZ,YYXYX},{YIIXI,ZIYYZ,IXZXZ,IIZXI,XYYYX},{YIZZI,ZIXII,YZYYI,ZIIXI,ZZIXX},{YIIXZ,XYIZZ,XZXYI,XXYYI,XXYIY},{IYXYI,YYZZZ,ZYZXI,ZZIZZ,YXYYX}}", "Q")


# write_mub_circuit_file("{{ZZYXZI,IIIIIZ,ZXXZYI,XYZXYI,YZXXZI,ZYXZXI},{XXXZXI,XZZYYI,XYYYYI,ZXXYXI,XYZYZI,IIIIIX},{YYZYYI,XZZYYZ,YZZXII,YZYZZI,ZXYZII,ZYXZXX},{XXXZXZ,YYYXII,IIXZII,XYIZYI,YIYXYI,ZZYXZY},{YIIIZI,IXIZII,ZXZZXI,IZIXII,YXXZIZ,ZXXZYY},{IZYYZI,ZIYZXI,YIZXZI,IYZZYZ,YZYXXZ,YZYYIX},{ZYIYYI,YXYXZI,YXIZXZ,IIIXZZ,ZXXIZI,ZXYIXX},{YZIIII,YIXZXZ,YZZXIZ,XYZIXI,YZXYYI,XXIYYX},{YYZYYZ,YYYXIZ,ZXIIYI,IIZYII,IZZIXI,IXZYYY},{YIIIZZ,ZIXIYI,YZIYZI,YIXIZI,XZIIXZ,IYZYXX},{ZYZXXI,YYXYZI,IZYIII,ZIYIZZ,XIIIYI,XYZXYY},{YIZZII,IYZIII,XZXIIZ,ZZYIII,IIIZXZ,XYIYXX},{IIXXZI,XYIIIZ,XIXIZI,YZYZZZ,IIZIYI,YIYYZX},{XIYXZZ,XXIIZI,IIXZIZ,YZXIII,ZYXIII,XIZIIY},{XZYXII,IXIZIZ,IIYIZI,XXZIYI,IYIYZZ,IZIIZY},{IZYYZZ,IXZIZI,ZYIIXI,YXYYXZ,XXZYIZ,XIIZZY},{IZXXII,ZZXIXI,IYXYYZ,ZYXYYZ,IYZXIZ,IIZZZY},{ZXZXYI,IZIYYZ,XXYYXZ,YXXXYZ,XYIXIZ,ZYYIYX},{IXYZXZ,XIZYXZ,IYYXXZ,ZXYXYZ,YIXYXI,ZIXYZY},{XYXZYZ,IZZXXZ,XYXXXZ,IZZYZI,YYYIIZ,XZZIZX},{IXXIYZ,XZIXXZ,YIIYII,IXIIYZ,IXIYII,XZIZIX},{XXYIYZ,YXXYII,YYZIXZ,YYYYYI,IXZXZI,XIZIIX},{YZZZZI,YZYIXZ,IXXYXI,YYXXXI,IYIYZI,ZYXZXY},{YXIXYZ,IIIYXI,IXYXYI,YXYYXI,YIYXYZ,IYZYXY},{IYYZYI,IIZXYI,IYXYYI,IZIXIZ,XIIIYZ,YZYYIY},{IYXIXI,IZIYYI,YIZXZZ,ZZYIIZ,ZXXIZZ,IIZZZX},{IXYZXI,YXYXZZ,XIXIZZ,XYZIXZ,XYIXII,IZZZIX},{YZIIIZ,XXIIZZ,ZXIIYZ,ZXYXYI,XXIXZI,YYYZXY},{XZYXIZ,ZIXIYZ,XYXXXI,ZYYXXI,ZIXXYZ,ZIYXIX},{ZYZXXZ,XZIXXI,XXXXYI,XZZXIZ,YYXZZI,YXXIXX},{XXYIYI,XIIXYI,ZIIXZZ,IXZZXI,ZZYYYI,YZXXZY},{XYYIXI,ZXXXZZ,YYIZYI,XIIYII,ZXYZIZ,IZIIZX},{ZZZIIZ,YZXZYI,ZZZYZI,XYIZYZ,XXZYII,YXXIXY},{YXZYXI,ZYYYZI,ZXZZXZ,ZYXYYI,ZZYYYZ,XYIYXY},{ZIIZII,ZIYZXZ,XXYYXI,XIIYIZ,IIZIYZ,XXZXXX},{ZYIYYZ,XIZYXI,ZZZYZZ,YZXIIZ,IZIZYI,IYIXYY},{XYXZYI,ZYYYZZ,IIYIZZ,YIYZII,XIZZXZ,YIXXIY},{ZIIZIZ,IXZIZZ,IZXZZI,ZZXZZZ,ZYYZZZ,YYXIYY},{IZXXIZ,IYIZZI,XIYZIZ,XXIZXZ,ZIYYXZ,ZZXYIX},{IIYYII,XXZZIZ,ZYZZYZ,XZIYZZ,YXYIZI,IZZZIY},{XZXYZZ,ZZYZYZ,ZIZYIZ,IYIIXI,XXIXZZ,XZZIZY},{ZXIYXZ,ZXYYIZ,YXZIYI,ZYYXXZ,IXIYIZ,YYXIYX},{ZZIZZZ,YIYIYI,XXXXYZ,YYYYYZ,ZIYYXI,IYIXYX},{YYIXXI,XIIXYZ,IXXYXZ,XZIYZI,XIZZXI,XXIYYY},{XYYIXZ,IIIYXZ,ZIZYII,ZZXZZI,IZZIXZ,ZIYXIY},{IYYZYZ,ZXYYII,XIYZII,YIXIZZ,YYXZZZ,XIIZZX},{ZZIZZI,XXZZII,IZYIIZ,IXZZXZ,IYZXII,ZXYIXY},{XZXYZI,IYZIIZ,YYIZYZ,YXXXYI,YZXYYZ,YIXXIX},{IIXXZZ,YZXZYZ,IYYXXI,IIZYIZ,ZYYZZI,XZIZIY},{YXZYXZ,IZZXXI,YZIYZZ,XXIZXI,IXZXZZ,YXYZYX},{IXXIYI,YYXYZZ,ZYZZYI,YYXXXZ,ZZXXXI,YIYYZY},{YIZZIZ,ZZYZYI,IXYXYZ,XIZXZI,ZYXIIZ,YXYZYY},{ZXIYXI,IIZXYZ,ZZIXII,XXZIYZ,ZZXXXZ,XXZXXY},{IYXIXZ,ZYXXII,ZYIIXZ,XIZXZZ,IZIZYZ,ZZXYIY},{ZIZIZI,ZZXIXZ,ZZIXIZ,YIYZIZ,YXYIZZ,ZYYIYY},{ZXZXYZ,ZYXXIZ,IZXZZZ,IYIIXZ,YIXYXZ,IXIXXY},{ZIZIZZ,IYIZZZ,YXZIYZ,IZZYZZ,XZZZYZ,IXIXXX},{IIYYIZ,YIYIYZ,YIIYIZ,ZIXZIZ,XZZZYI,ZIXYZX},{YYIXXZ,YXXYIZ,XZYZZZ,ZIXZII,YYYIII,YYYZXX},{YZZZZZ,XYZZZZ,XZYZZI,IXIIYI,ZIXXYI,IXZYYX},{XIXYIZ,XYZZZI,YYZIXI,XZZXII,XZIIXI,ZZYXZX},{XIXYII,YZYIXI,ZIIXZI,ZIYIZI,YXXZII,IIIIIY},{YXIXYI,ZXXXZI,XZXIII,IYZZYI,XYZYZZ,ZXXZYX},{ZZZIII,XYIIII,YXIZXI,ZXXYXZ,YZYXXI,XYZXYX},{XIYXZI,YIXZXI,XYYYYZ,IIIXZI,IIIZXI,YZXXZX}}", "all")
