"""Read-only bytecode scan of the vanilla 1.21.4 server jar (in memory).

Finds the light-update packet class through its registry name, then every
method that constructs it, and prints the instructions around each `new`,
so we can see which player set the server sends light updates to.
"""
import io
import struct
import sys
import zipfile

from pathlib import Path

# The jar setup.cmd installs; pass another path as the first argument.
JAR = sys.argv[1] if len(sys.argv) > 1 else str(
    Path(__file__).resolve().parents[3] / ".local" / "server" / "server-1.21.4.jar")

outer = zipfile.ZipFile(JAR)
inner = zipfile.ZipFile(io.BytesIO(outer.read("META-INF/versions/1.21.4/server-1.21.4.jar")))


class ClassFile:
    def __init__(self, data):
        self.data = data
        self.pos = 8
        n = self.u2()
        self.cp = [None] * n
        i = 1
        while i < n:
            tag = self.u1()
            if tag == 1:
                length = self.u2()
                self.cp[i] = ("Utf8", self.data[self.pos:self.pos + length].decode("utf-8", "replace"))
                self.pos += length
            elif tag in (3, 4):
                self.cp[i] = ("Int", self.data[self.pos:self.pos + 4]); self.pos += 4
            elif tag in (5, 6):
                self.cp[i] = ("Long", self.data[self.pos:self.pos + 8]); self.pos += 8; i += 1
            elif tag == 7:
                self.cp[i] = ("Class", self.u2())
            elif tag == 8:
                self.cp[i] = ("String", self.u2())
            elif tag in (9, 10, 11):
                self.cp[i] = ({9: "Field", 10: "Method", 11: "IMethod"}[tag], self.u2(), self.u2())
            elif tag == 12:
                self.cp[i] = ("NameAndType", self.u2(), self.u2())
            elif tag == 15:
                self.cp[i] = ("MethodHandle", self.u1(), self.u2())
            elif tag == 16:
                self.cp[i] = ("MethodType", self.u2())
            elif tag in (17, 18):
                self.cp[i] = ("Dynamic", self.u2(), self.u2())
            elif tag in (19, 20):
                self.cp[i] = ("Module", self.u2())
            else:
                raise ValueError(f"bad tag {tag}")
            i += 1
        self.access = self.u2()
        self.this = self.class_name(self.u2())
        self.super = self.u2()
        for _ in range(self.u2()):
            self.u2()
        self.fields = [self.member() for _ in range(self.u2())]
        self.methods = [self.member() for _ in range(self.u2())]

    def u1(self):
        v = self.data[self.pos]; self.pos += 1; return v

    def u2(self):
        v = struct.unpack(">H", self.data[self.pos:self.pos + 2])[0]; self.pos += 2; return v

    def u4(self):
        v = struct.unpack(">I", self.data[self.pos:self.pos + 4])[0]; self.pos += 4; return v

    def utf(self, i):
        return self.cp[i][1]

    def class_name(self, i):
        return self.utf(self.cp[i][1])

    def member(self):
        access, name, desc = self.u2(), self.u2(), self.u2()
        code = None
        for _ in range(self.u2()):
            aname = self.utf(self.u2())
            length = self.u4()
            body = self.data[self.pos:self.pos + length]
            self.pos += length
            if aname == "Code":
                code_len = struct.unpack(">I", body[4:8])[0]
                code = body[8:8 + code_len]
        return {"name": self.utf(name), "desc": self.utf(desc), "code": code}

    def ref(self, i):
        e = self.cp[i]
        if e[0] in ("Field", "Method", "IMethod"):
            nat = self.cp[e[2]]
            return f"{self.class_name(e[1])}.{self.utf(nat[1])}{self.utf(nat[2])}"
        if e[0] == "Class":
            return self.class_name(i)
        if e[0] == "String":
            return repr(self.utf(e[1]))
        return str(e)


# Opcode lengths (bytes after the opcode) for the fixed-size instructions.
LEN = {o: 0 for o in range(256)}
for o in (0x10, 0x12, 0x15, 0x16, 0x17, 0x18, 0x19, 0x36, 0x37, 0x38, 0x39, 0x3a, 0xa9, 0xbc):
    LEN[o] = 1
for o in (0x11, 0x13, 0x14, 0x84, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xbb, 0xbd, 0xc0, 0xc1, 0xc6, 0xc7) + tuple(range(0x99, 0xa9)):
    LEN[o] = 2
LEN[0xc5] = 3
for o in (0xb9, 0xba, 0xc8, 0xc9):
    LEN[o] = 4
NAMES = {0x03: "iconst_0", 0x04: "iconst_1", 0xbb: "new", 0xb6: "invokevirtual", 0xb7: "invokespecial", 0xb8: "invokestatic",
         0xb9: "invokeinterface", 0xb2: "getstatic", 0xb3: "putstatic", 0xb4: "getfield", 0xb5: "putfield", 0x12: "ldc", 0x13: "ldc_w",
         0x2a: "aload_0", 0x2b: "aload_1", 0x2c: "aload_2", 0x2d: "aload_3", 0x19: "aload", 0x3a: "astore", 0x4c: "astore_1", 0x4d: "astore_2",
         0x4e: "astore_3", 0xc6: "ifnull", 0xc7: "ifnonnull", 0x99: "ifeq", 0x9a: "ifne", 0xa7: "goto"}


def instructions(cf, code):
    pc = 0
    while pc < len(code):
        op = code[pc]
        if op == 0xaa:  # tableswitch
            p = (pc + 4) & ~3
            low, high = struct.unpack(">ii", code[p + 4:p + 12])
            size = p + 12 + 4 * (high - low + 1) - pc
            yield pc, "tableswitch", None
            pc += size
            continue
        if op == 0xab:  # lookupswitch
            p = (pc + 4) & ~3
            n = struct.unpack(">i", code[p + 4:p + 8])[0]
            yield pc, "lookupswitch", None
            pc += p + 8 + 8 * n - pc
            continue
        if op == 0xc4:  # wide
            inner_op = code[pc + 1]
            size = 6 if inner_op == 0x84 else 4
            yield pc, "wide", None
            pc += size
            continue
        operand = None
        if op in (0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xbb, 0xbd, 0xc0, 0xc1, 0x13, 0x14):
            idx = struct.unpack(">H", code[pc + 1:pc + 3])[0]
            operand = cf.ref(idx) if cf.cp[idx] and cf.cp[idx][0] not in ("Int", "Long") else idx
        elif op == 0x12:
            idx = code[pc + 1]
            operand = cf.ref(idx) if cf.cp[idx] and cf.cp[idx][0] not in ("Int", "Long") else idx
        yield pc, NAMES.get(op, hex(op)), operand
        pc += 1 + LEN[op]


classes = {}
for name in inner.namelist():
    if name.endswith(".class"):
        try:
            cf = ClassFile(inner.read(name))
        except Exception as error:  # noqa: BLE001
            continue
        classes[cf.this] = cf
print("classes parsed:", len(classes), file=sys.stderr)

# 1. The packet-type registry field named "light_update".
holders = []
for cname, cf in classes.items():
    for m in cf.methods:
        if not m["code"]:
            continue
        ins = list(instructions(cf, m["code"]))
        for k, (pc, op, arg) in enumerate(ins):
            if op in ("ldc", "ldc_w") and arg == "'light_update'":
                for pc2, op2, arg2 in ins[k:k + 4]:
                    if op2 == "putstatic":
                        holders.append(arg2)
print("light_update packet type fields:", holders)
field_refs = {h.split("L")[0] if False else h for h in holders}

# 2. Classes that read that field: the packet class's type() method.
packet_classes = set()
for cname, cf in classes.items():
    for m in cf.methods:
        if not m["code"]:
            continue
        for pc, op, arg in instructions(cf, m["code"]):
            if op == "getstatic" and arg in field_refs and m["desc"].startswith("()"):
                packet_classes.add((cname, m["name"], m["desc"]))
print("readers of the field (packet type() methods):", packet_classes)
packet_names = {c for c, _, _ in packet_classes}

# 3. Every method that constructs the packet, with the instructions before the `new` and after.
for cname, cf in classes.items():
    for m in cf.methods:
        if not m["code"]:
            continue
        ins = list(instructions(cf, m["code"]))
        for k, (pc, op, arg) in enumerate(ins):
            if op == "new" and arg in packet_names:
                print(f"\n== {cname}.{m['name']}{m['desc']} constructs {arg} at pc {pc}")
                for pc2, op2, arg2 in ins[max(0, k - 14):k + 14]:
                    print(f"   {pc2:5d} {op2:16s} {arg2 if arg2 is not None else ''}")
