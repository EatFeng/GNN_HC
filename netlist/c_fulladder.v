// Verilog
// c_fulladder
// Ninputs 3
// Noutputs 2
// 12 gates (8 NANDs + 3 Not + 1 Nor)

module c_fulladder (Cin, A, B, Carry, Sum);

input Cin, A, B;

output Carry, Sum;

wire N1, N2, N3, N4, N5, N6, N8, N9, N10, N11;

nand NAND2_1 (N1, A, B);
nand NAND2_2 (N2, N1, A);
nand NAND2_3 (N3, N1, B);
nand NAND2_4 (N4, N2, N3);
not NOT1_1 (N10, N1);
nand NAND2_5 (N5, Cin, N4);
nand NAND2_6 (N6, Cin, N5);
nand NAND2_7 (Sum, N6, N8);
nand NAND2_8 (N8, N5, N4);
not NOT1_2 (N9, N5);
nor NOR2_1 (N11, N9, N10);
not NOT1_3 (Carry, N11);

endmodule
