// Verilog
// c_halfadder
// Ninputs 2
// Noutputs 2
// 5 gates (4 NANDs + 1 Not)

module c_halfadder (N1,N2, Carry, Sum);

input N1,N2;

output Carry, Sum;

wire N3, N4, N5;

nand NAND2_1 (N3, N1, N2);
nand NAND2_2 (N4, N1, N3);
nand NAND2_3 (N5, N2, N3);
nand NAND2_4 (Sum, N4, N5);
not NOT1_5 (Carry, N3);

endmodule
