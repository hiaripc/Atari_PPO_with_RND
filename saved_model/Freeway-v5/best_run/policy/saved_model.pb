╗Є
с░
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
┴
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
ў
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e58ТТ

v
critic_int/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namecritic_int/bias
o
#critic_int/bias/Read/ReadVariableOpReadVariableOpcritic_int/bias*
_output_shapes
:*
dtype0

critic_int/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*"
shared_namecritic_int/kernel
x
%critic_int/kernel/Read/ReadVariableOpReadVariableOpcritic_int/kernel*
_output_shapes
:	А*
dtype0
v
critic_ext/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namecritic_ext/bias
o
#critic_ext/bias/Read/ReadVariableOpReadVariableOpcritic_ext/bias*
_output_shapes
:*
dtype0

critic_ext/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*"
shared_namecritic_ext/kernel
x
%critic_ext/kernel/Read/ReadVariableOpReadVariableOpcritic_ext/kernel*
_output_shapes
:	А*
dtype0
l

actor/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
actor/bias
e
actor/bias/Read/ReadVariableOpReadVariableOp
actor/bias*
_output_shapes
:*
dtype0
u
actor/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*
shared_nameactor/kernel
n
 actor/kernel/Read/ReadVariableOpReadVariableOpactor/kernel*
_output_shapes
:	А*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:А*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АbА*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
АbА*
dtype0
r
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_5/bias
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
:@*
dtype0
В
conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_5/kernel
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
:@*
dtype0
В
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
: @*
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
: *
dtype0
В
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
: *
dtype0
Т
serving_default_input_2Placeholder*3
_output_shapes!
:         TT*
dtype0*(
shape:         TT
╫
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2conv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biasdense_1/kerneldense_1/biascritic_int/kernelcritic_int/biascritic_ext/kernelcritic_ext/biasactor/kernel
actor/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *+
f&R$
"__inference_signature_wrapper_1275

NoOpNoOp
и8
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*у7
value┘7B╓7 B╧7
ш
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
╚
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op*
╚
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses

!kernel
"bias
 #_jit_compiled_convolution_op*
╚
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses

*kernel
+bias
 ,_jit_compiled_convolution_op*
О
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses* 
ж
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses

9kernel
:bias*
ж
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses

Akernel
Bbias*
ж
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses

Ikernel
Jbias*
ж
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses

Qkernel
Rbias*
j
0
1
!2
"3
*4
+5
96
:7
A8
B9
I10
J11
Q12
R13*
j
0
1
!2
"3
*4
+5
96
:7
A8
B9
I10
J11
Q12
R13*
* 
░
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Xtrace_0
Ytrace_1
Ztrace_2
[trace_3* 
6
\trace_0
]trace_1
^trace_2
_trace_3* 
* 

`serving_default* 

0
1*

0
1*
* 
У
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

ftrace_0* 

gtrace_0* 
_Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

!0
"1*

!0
"1*
* 
У
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses*

mtrace_0* 

ntrace_0* 
_Y
VARIABLE_VALUEconv2d_4/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_4/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

*0
+1*

*0
+1*
* 
У
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses*

ttrace_0* 

utrace_0* 
_Y
VARIABLE_VALUEconv2d_5/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_5/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
С
vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses* 

{trace_0* 

|trace_0* 

90
:1*

90
:1*
* 
Х
}non_trainable_variables

~layers
metrics
 Аlayer_regularization_losses
Бlayer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses*

Вtrace_0* 

Гtrace_0* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

A0
B1*

A0
B1*
* 
Ш
Дnon_trainable_variables
Еlayers
Жmetrics
 Зlayer_regularization_losses
Иlayer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses*

Йtrace_0* 

Кtrace_0* 
\V
VARIABLE_VALUEactor/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
actor/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

I0
J1*

I0
J1*
* 
Ш
Лnon_trainable_variables
Мlayers
Нmetrics
 Оlayer_regularization_losses
Пlayer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses*

Рtrace_0* 

Сtrace_0* 
a[
VARIABLE_VALUEcritic_ext/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEcritic_ext/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

Q0
R1*

Q0
R1*
* 
Ш
Тnon_trainable_variables
Уlayers
Фmetrics
 Хlayer_regularization_losses
Цlayer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses*

Чtrace_0* 

Шtrace_0* 
a[
VARIABLE_VALUEcritic_int/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEcritic_int/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
C
0
1
2
3
4
5
6
7
	8*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
в
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp actor/kernel/Read/ReadVariableOpactor/bias/Read/ReadVariableOp%critic_ext/kernel/Read/ReadVariableOp#critic_ext/bias/Read/ReadVariableOp%critic_int/kernel/Read/ReadVariableOp#critic_int/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *&
f!R
__inference__traced_save_1873
Е
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biasdense_1/kerneldense_1/biasactor/kernel
actor/biascritic_ext/kernelcritic_ext/biascritic_int/kernelcritic_int/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *)
f$R"
 __inference__traced_restore_1925фо	
╩	
ї
C__inference_critic_ext_layer_call_and_return_conditional_losses_862

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╠
^
B__inference_flatten_1_layer_call_and_return_conditional_losses_817

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"     1  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         АbY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         Аb"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @:[ W
3
_output_shapes!
:         @
 
_user_specified_nameinputs
╪*
╡
?__inference_policy_layer_call_and_return_conditional_losses_888

inputs&
conv2d_3_728: 
conv2d_3_730: &
conv2d_4_767: @
conv2d_4_769:@&
conv2d_5_806:@@
conv2d_5_808:@
dense_1_831:
АbА
dense_1_833:	А!
critic_int_847:	А
critic_int_849:!
critic_ext_863:	А
critic_ext_865:
	actor_880:	А
	actor_882:
identity

identity_1

identity_2Ивactor/StatefulPartitionedCallв conv2d_3/StatefulPartitionedCallв conv2d_4/StatefulPartitionedCallв conv2d_5/StatefulPartitionedCallв"critic_ext/StatefulPartitionedCallв"critic_int/StatefulPartitionedCallвdense_1/StatefulPartitionedCallЎ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_3_728conv2d_3_730*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_conv2d_3_layer_call_and_return_conditional_losses_727Щ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv2d_4_767conv2d_4_769*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_conv2d_4_layer_call_and_return_conditional_losses_766Щ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0conv2d_5_806conv2d_5_808*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_conv2d_5_layer_call_and_return_conditional_losses_805▐
flatten_1/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Аb* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_flatten_1_layer_call_and_return_conditional_losses_817Г
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_831dense_1_833*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_830Ф
"critic_int/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0critic_int_847critic_int_849*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_critic_int_layer_call_and_return_conditional_losses_846Ф
"critic_ext/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0critic_ext_863critic_ext_865*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_critic_ext_layer_call_and_return_conditional_losses_862А
actor/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0	actor_880	actor_882*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *G
fBR@
>__inference_actor_layer_call_and_return_conditional_losses_879u
IdentityIdentity&actor/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         |

Identity_1Identity+critic_ext/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         |

Identity_2Identity+critic_int/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ╗
NoOpNoOp^actor/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall#^critic_ext/StatefulPartitionedCall#^critic_int/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         TT: : : : : : : : : : : : : : 2>
actor/StatefulPartitionedCallactor/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2H
"critic_ext/StatefulPartitionedCall"critic_ext/StatefulPartitionedCall2H
"critic_int/StatefulPartitionedCall"critic_int/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:[ W
3
_output_shapes!
:         TT
 
_user_specified_nameinputs
б%
о
A__inference_conv2d_5_layer_call_and_return_conditional_losses_805

inputs?
%conv2d_conv2d_readvariableop_resource:@@@
2squeeze_batch_dims_biasadd_readvariableop_resource:@
identityИвConv2D/Conv2D/ReadVariableOpв)squeeze_batch_dims/BiasAdd/ReadVariableOpB
Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:d
Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
¤        f
Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
Conv2D/strided_sliceStridedSliceConv2D/Shape:output:0#Conv2D/strided_slice/stack:output:0%Conv2D/strided_slice/stack_1:output:0%Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    	   	   @   z
Conv2D/ReshapeReshapeinputsConv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:         		@К
Conv2D/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0╣
Conv2D/Conv2DConv2DConv2D/Reshape:output:0$Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
k
Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      @   ]
Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         д
Conv2D/concatConcatV2Conv2D/strided_slice:output:0Conv2D/concat/values_1:output:0Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:Й
Conv2D/Reshape_1ReshapeConv2D/Conv2D:output:0Conv2D/concat:output:0*
T0*3
_output_shapes!
:         @a
squeeze_batch_dims/ShapeShapeConv2D/Reshape_1:output:0*
T0*
_output_shapes
:p
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
¤        r
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:о
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masky
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"          @   е
squeeze_batch_dims/ReshapeReshapeConv2D/Reshape_1:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:         @Ш
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╖
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @w
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      @   i
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         ╘
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:о
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:         @q
ReluRelu%squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:         @m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:         @С
NoOpNoOp^Conv2D/Conv2D/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         		@: : 2<
Conv2D/Conv2D/ReadVariableOpConv2D/Conv2D/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:[ W
3
_output_shapes!
:         		@
 
_user_specified_nameinputs
в%
п
B__inference_conv2d_3_layer_call_and_return_conditional_losses_1633

inputs?
%conv2d_conv2d_readvariableop_resource: @
2squeeze_batch_dims_biasadd_readvariableop_resource: 
identityИвConv2D/Conv2D/ReadVariableOpв)squeeze_batch_dims/BiasAdd/ReadVariableOpB
Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:d
Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
¤        f
Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
Conv2D/strided_sliceStridedSliceConv2D/Shape:output:0#Conv2D/strided_slice/stack:output:0%Conv2D/strided_slice/stack_1:output:0%Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    T   T      z
Conv2D/ReshapeReshapeinputsConv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:         TTК
Conv2D/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0╣
Conv2D/Conv2DConv2DConv2D/Reshape:output:0$Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingVALID*
strides
k
Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"          ]
Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         д
Conv2D/concatConcatV2Conv2D/strided_slice:output:0Conv2D/concat/values_1:output:0Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:Й
Conv2D/Reshape_1ReshapeConv2D/Conv2D:output:0Conv2D/concat:output:0*
T0*3
_output_shapes!
:          a
squeeze_batch_dims/ShapeShapeConv2D/Reshape_1:output:0*
T0*
_output_shapes
:p
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
¤        r
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:о
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masky
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"              е
squeeze_batch_dims/ReshapeReshapeConv2D/Reshape_1:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:          Ш
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0╖
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          w
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"          i
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         ╘
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:о
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:          q
ReluRelu%squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:          m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:          С
NoOpNoOp^Conv2D/Conv2D/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         TT: : 2<
Conv2D/Conv2D/ReadVariableOpConv2D/Conv2D/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:[ W
3
_output_shapes!
:         TT
 
_user_specified_nameinputs
╟
Ч
)__inference_critic_int_layer_call_fn_1796

inputs
unknown:	А
	unknown_0:
identityИвStatefulPartitionedCall█
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_critic_int_layer_call_and_return_conditional_losses_846o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Ю

Ё
>__inference_actor_layer_call_and_return_conditional_losses_879

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╜
Т
$__inference_actor_layer_call_fn_1757

inputs
unknown:	А
	unknown_0:
identityИвStatefulPartitionedCall╓
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *G
fBR@
>__inference_actor_layer_call_and_return_conditional_losses_879o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
д

ї
A__inference_dense_1_layer_call_and_return_conditional_losses_1748

inputs2
matmul_readvariableop_resource:
АbА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АbА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         Аw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Аb: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         Аb
 
_user_specified_nameinputs
╦	
Ў
D__inference_critic_ext_layer_call_and_return_conditional_losses_1787

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
ш
Ш
%__inference_policy_layer_call_fn_1152
input_2!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:
АbА
	unknown_6:	А
	unknown_7:	А
	unknown_8:
	unknown_9:	А

unknown_10:

unknown_11:	А

unknown_12:
identity

identity_1

identity_2ИвStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_policy_layer_call_and_return_conditional_losses_1080o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         TT: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:         TT
!
_user_specified_name	input_2
ф
Ч
%__inference_policy_layer_call_fn_1312

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:
АbА
	unknown_6:	А
	unknown_7:	А
	unknown_8:
	unknown_9:	А

unknown_10:

unknown_11:	А

unknown_12:
identity

identity_1

identity_2ИвStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_policy_layer_call_and_return_conditional_losses_888o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         TT: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:         TT
 
_user_specified_nameinputs
а'
№
__inference__traced_save_1873
file_prefix.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop.
*savev2_conv2d_5_kernel_read_readvariableop,
(savev2_conv2d_5_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop+
'savev2_actor_kernel_read_readvariableop)
%savev2_actor_bias_read_readvariableop0
,savev2_critic_ext_kernel_read_readvariableop.
*savev2_critic_ext_bias_read_readvariableop0
,savev2_critic_int_kernel_read_readvariableop.
*savev2_critic_int_bias_read_readvariableop
savev2_const

identity_1ИвMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: М
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*╡
valueлBиB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЛ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B └
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop'savev2_actor_kernel_read_readvariableop%savev2_actor_bias_read_readvariableop,savev2_critic_ext_kernel_read_readvariableop*savev2_critic_ext_bias_read_readvariableop,savev2_critic_int_kernel_read_readvariableop*savev2_critic_int_bias_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*з
_input_shapesХ
Т: : : : @:@:@@:@:
АbА:А:	А::	А::	А:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:&"
 
_output_shapes
:
АbА:!

_output_shapes	
:А:%	!

_output_shapes
:	А: 


_output_shapes
::%!

_output_shapes
:	А: 

_output_shapes
::%!

_output_shapes
:	А: 

_output_shapes
::

_output_shapes
: 
ї*
─
@__inference_policy_layer_call_and_return_conditional_losses_1080

inputs'
conv2d_3_1041: 
conv2d_3_1043: '
conv2d_4_1046: @
conv2d_4_1048:@'
conv2d_5_1051:@@
conv2d_5_1053:@ 
dense_1_1057:
АbА
dense_1_1059:	А"
critic_int_1062:	А
critic_int_1064:"
critic_ext_1067:	А
critic_ext_1069:

actor_1072:	А

actor_1074:
identity

identity_1

identity_2Ивactor/StatefulPartitionedCallв conv2d_3/StatefulPartitionedCallв conv2d_4/StatefulPartitionedCallв conv2d_5/StatefulPartitionedCallв"critic_ext/StatefulPartitionedCallв"critic_int/StatefulPartitionedCallвdense_1/StatefulPartitionedCall°
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_3_1041conv2d_3_1043*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_conv2d_3_layer_call_and_return_conditional_losses_727Ы
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv2d_4_1046conv2d_4_1048*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_conv2d_4_layer_call_and_return_conditional_losses_766Ы
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0conv2d_5_1051conv2d_5_1053*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_conv2d_5_layer_call_and_return_conditional_losses_805▐
flatten_1/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Аb* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_flatten_1_layer_call_and_return_conditional_losses_817Е
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_1057dense_1_1059*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_830Ц
"critic_int/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0critic_int_1062critic_int_1064*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_critic_int_layer_call_and_return_conditional_losses_846Ц
"critic_ext/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0critic_ext_1067critic_ext_1069*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_critic_ext_layer_call_and_return_conditional_losses_862В
actor/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0
actor_1072
actor_1074*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *G
fBR@
>__inference_actor_layer_call_and_return_conditional_losses_879u
IdentityIdentity&actor/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         |

Identity_1Identity+critic_ext/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         |

Identity_2Identity+critic_int/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ╗
NoOpNoOp^actor/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall#^critic_ext/StatefulPartitionedCall#^critic_int/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         TT: : : : : : : : : : : : : : 2>
actor/StatefulPartitionedCallactor/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2H
"critic_ext/StatefulPartitionedCall"critic_ext/StatefulPartitionedCall2H
"critic_int/StatefulPartitionedCall"critic_int/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:[ W
3
_output_shapes!
:         TT
 
_user_specified_nameinputs
б%
о
A__inference_conv2d_4_layer_call_and_return_conditional_losses_766

inputs?
%conv2d_conv2d_readvariableop_resource: @@
2squeeze_batch_dims_biasadd_readvariableop_resource:@
identityИвConv2D/Conv2D/ReadVariableOpв)squeeze_batch_dims/BiasAdd/ReadVariableOpB
Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:d
Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
¤        f
Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
Conv2D/strided_sliceStridedSliceConv2D/Shape:output:0#Conv2D/strided_slice/stack:output:0%Conv2D/strided_slice/stack_1:output:0%Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"              z
Conv2D/ReshapeReshapeinputsConv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:          К
Conv2D/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0╣
Conv2D/Conv2DConv2DConv2D/Reshape:output:0$Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         		@*
paddingVALID*
strides
k
Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"	   	   @   ]
Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         д
Conv2D/concatConcatV2Conv2D/strided_slice:output:0Conv2D/concat/values_1:output:0Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:Й
Conv2D/Reshape_1ReshapeConv2D/Conv2D:output:0Conv2D/concat:output:0*
T0*3
_output_shapes!
:         		@a
squeeze_batch_dims/ShapeShapeConv2D/Reshape_1:output:0*
T0*
_output_shapes
:p
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
¤        r
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:о
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masky
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    	   	   @   е
squeeze_batch_dims/ReshapeReshapeConv2D/Reshape_1:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:         		@Ш
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╖
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         		@w
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"	   	   @   i
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         ╘
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:о
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:         		@q
ReluRelu%squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:         		@m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:         		@С
NoOpNoOp^Conv2D/Conv2D/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:          : : 2<
Conv2D/Conv2D/ReadVariableOpConv2D/Conv2D/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:[ W
3
_output_shapes!
:          
 
_user_specified_nameinputs
в%
п
B__inference_conv2d_5_layer_call_and_return_conditional_losses_1717

inputs?
%conv2d_conv2d_readvariableop_resource:@@@
2squeeze_batch_dims_biasadd_readvariableop_resource:@
identityИвConv2D/Conv2D/ReadVariableOpв)squeeze_batch_dims/BiasAdd/ReadVariableOpB
Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:d
Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
¤        f
Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
Conv2D/strided_sliceStridedSliceConv2D/Shape:output:0#Conv2D/strided_slice/stack:output:0%Conv2D/strided_slice/stack_1:output:0%Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    	   	   @   z
Conv2D/ReshapeReshapeinputsConv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:         		@К
Conv2D/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0╣
Conv2D/Conv2DConv2DConv2D/Reshape:output:0$Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
k
Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      @   ]
Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         д
Conv2D/concatConcatV2Conv2D/strided_slice:output:0Conv2D/concat/values_1:output:0Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:Й
Conv2D/Reshape_1ReshapeConv2D/Conv2D:output:0Conv2D/concat:output:0*
T0*3
_output_shapes!
:         @a
squeeze_batch_dims/ShapeShapeConv2D/Reshape_1:output:0*
T0*
_output_shapes
:p
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
¤        r
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:о
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masky
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"          @   е
squeeze_batch_dims/ReshapeReshapeConv2D/Reshape_1:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:         @Ш
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╖
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @w
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      @   i
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         ╘
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:о
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:         @q
ReluRelu%squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:         @m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:         @С
NoOpNoOp^Conv2D/Conv2D/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         		@: : 2<
Conv2D/Conv2D/ReadVariableOpConv2D/Conv2D/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:[ W
3
_output_shapes!
:         		@
 
_user_specified_nameinputs
°
Ь
'__inference_conv2d_3_layer_call_fn_1600

inputs!
unknown: 
	unknown_0: 
identityИвStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_conv2d_3_layer_call_and_return_conditional_losses_727{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         TT: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:         TT
 
_user_specified_nameinputs
°
Ь
'__inference_conv2d_5_layer_call_fn_1684

inputs!
unknown:@@
	unknown_0:@
identityИвStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_conv2d_5_layer_call_and_return_conditional_losses_805{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         		@: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:         		@
 
_user_specified_nameinputs
═
_
C__inference_flatten_1_layer_call_and_return_conditional_losses_1728

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"     1  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         АbY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         Аb"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @:[ W
3
_output_shapes!
:         @
 
_user_specified_nameinputs
ц
Ч
$__inference_policy_layer_call_fn_923
input_2!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:
АbА
	unknown_6:	А
	unknown_7:	А
	unknown_8:
	unknown_9:	А

unknown_10:

unknown_11:	А

unknown_12:
identity

identity_1

identity_2ИвStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_policy_layer_call_and_return_conditional_losses_888o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         TT: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:         TT
!
_user_specified_name	input_2
г

Ї
@__inference_dense_1_layer_call_and_return_conditional_losses_830

inputs2
matmul_readvariableop_resource:
АbА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АbА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         Аw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Аb: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         Аb
 
_user_specified_nameinputs
в%
п
B__inference_conv2d_4_layer_call_and_return_conditional_losses_1675

inputs?
%conv2d_conv2d_readvariableop_resource: @@
2squeeze_batch_dims_biasadd_readvariableop_resource:@
identityИвConv2D/Conv2D/ReadVariableOpв)squeeze_batch_dims/BiasAdd/ReadVariableOpB
Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:d
Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
¤        f
Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
Conv2D/strided_sliceStridedSliceConv2D/Shape:output:0#Conv2D/strided_slice/stack:output:0%Conv2D/strided_slice/stack_1:output:0%Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"              z
Conv2D/ReshapeReshapeinputsConv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:          К
Conv2D/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0╣
Conv2D/Conv2DConv2DConv2D/Reshape:output:0$Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         		@*
paddingVALID*
strides
k
Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"	   	   @   ]
Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         д
Conv2D/concatConcatV2Conv2D/strided_slice:output:0Conv2D/concat/values_1:output:0Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:Й
Conv2D/Reshape_1ReshapeConv2D/Conv2D:output:0Conv2D/concat:output:0*
T0*3
_output_shapes!
:         		@a
squeeze_batch_dims/ShapeShapeConv2D/Reshape_1:output:0*
T0*
_output_shapes
:p
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
¤        r
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:о
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masky
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    	   	   @   е
squeeze_batch_dims/ReshapeReshapeConv2D/Reshape_1:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:         		@Ш
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╖
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         		@w
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"	   	   @   i
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         ╘
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:о
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:         		@q
ReluRelu%squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:         		@m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:         		@С
NoOpNoOp^Conv2D/Conv2D/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:          : : 2<
Conv2D/Conv2D/ReadVariableOpConv2D/Conv2D/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:[ W
3
_output_shapes!
:          
 
_user_specified_nameinputs
╦	
Ў
D__inference_critic_int_layer_call_and_return_conditional_losses_1806

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
б%
о
A__inference_conv2d_3_layer_call_and_return_conditional_losses_727

inputs?
%conv2d_conv2d_readvariableop_resource: @
2squeeze_batch_dims_biasadd_readvariableop_resource: 
identityИвConv2D/Conv2D/ReadVariableOpв)squeeze_batch_dims/BiasAdd/ReadVariableOpB
Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:d
Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
¤        f
Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
Conv2D/strided_sliceStridedSliceConv2D/Shape:output:0#Conv2D/strided_slice/stack:output:0%Conv2D/strided_slice/stack_1:output:0%Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    T   T      z
Conv2D/ReshapeReshapeinputsConv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:         TTК
Conv2D/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0╣
Conv2D/Conv2DConv2DConv2D/Reshape:output:0$Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingVALID*
strides
k
Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"          ]
Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         д
Conv2D/concatConcatV2Conv2D/strided_slice:output:0Conv2D/concat/values_1:output:0Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:Й
Conv2D/Reshape_1ReshapeConv2D/Conv2D:output:0Conv2D/concat:output:0*
T0*3
_output_shapes!
:          a
squeeze_batch_dims/ShapeShapeConv2D/Reshape_1:output:0*
T0*
_output_shapes
:p
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
¤        r
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:о
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masky
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"              е
squeeze_batch_dims/ReshapeReshapeConv2D/Reshape_1:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:          Ш
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0╖
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          w
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"          i
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         ╘
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:о
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:          q
ReluRelu%squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:          m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:          С
NoOpNoOp^Conv2D/Conv2D/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         TT: : 2<
Conv2D/Conv2D/ReadVariableOpConv2D/Conv2D/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:[ W
3
_output_shapes!
:         TT
 
_user_specified_nameinputs
┼
Ц
&__inference_dense_1_layer_call_fn_1737

inputs
unknown:
АbА
	unknown_0:	А
identityИвStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_830p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Аb: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         Аb
 
_user_specified_nameinputs
°*
┼
@__inference_policy_layer_call_and_return_conditional_losses_1236
input_2'
conv2d_3_1197: 
conv2d_3_1199: '
conv2d_4_1202: @
conv2d_4_1204:@'
conv2d_5_1207:@@
conv2d_5_1209:@ 
dense_1_1213:
АbА
dense_1_1215:	А"
critic_int_1218:	А
critic_int_1220:"
critic_ext_1223:	А
critic_ext_1225:

actor_1228:	А

actor_1230:
identity

identity_1

identity_2Ивactor/StatefulPartitionedCallв conv2d_3/StatefulPartitionedCallв conv2d_4/StatefulPartitionedCallв conv2d_5/StatefulPartitionedCallв"critic_ext/StatefulPartitionedCallв"critic_int/StatefulPartitionedCallвdense_1/StatefulPartitionedCall∙
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_3_1197conv2d_3_1199*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_conv2d_3_layer_call_and_return_conditional_losses_727Ы
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv2d_4_1202conv2d_4_1204*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_conv2d_4_layer_call_and_return_conditional_losses_766Ы
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0conv2d_5_1207conv2d_5_1209*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_conv2d_5_layer_call_and_return_conditional_losses_805▐
flatten_1/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Аb* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_flatten_1_layer_call_and_return_conditional_losses_817Е
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_1213dense_1_1215*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_830Ц
"critic_int/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0critic_int_1218critic_int_1220*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_critic_int_layer_call_and_return_conditional_losses_846Ц
"critic_ext/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0critic_ext_1223critic_ext_1225*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_critic_ext_layer_call_and_return_conditional_losses_862В
actor/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0
actor_1228
actor_1230*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *G
fBR@
>__inference_actor_layer_call_and_return_conditional_losses_879u
IdentityIdentity&actor/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         |

Identity_1Identity+critic_ext/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         |

Identity_2Identity+critic_int/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ╗
NoOpNoOp^actor/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall#^critic_ext/StatefulPartitionedCall#^critic_int/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         TT: : : : : : : : : : : : : : 2>
actor/StatefulPartitionedCallactor/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2H
"critic_ext/StatefulPartitionedCall"critic_ext/StatefulPartitionedCall2H
"critic_int/StatefulPartitionedCall"critic_int/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:\ X
3
_output_shapes!
:         TT
!
_user_specified_name	input_2
├
Х
"__inference_signature_wrapper_1275
input_2!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:
АbА
	unknown_6:	А
	unknown_7:	А
	unknown_8:
	unknown_9:	А

unknown_10:

unknown_11:	А

unknown_12:
identity

identity_1

identity_2ИвStatefulPartitionedCall■
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *'
f"R 
__inference__wrapped_model_687o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         TT: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:         TT
!
_user_specified_name	input_2
°*
┼
@__inference_policy_layer_call_and_return_conditional_losses_1194
input_2'
conv2d_3_1155: 
conv2d_3_1157: '
conv2d_4_1160: @
conv2d_4_1162:@'
conv2d_5_1165:@@
conv2d_5_1167:@ 
dense_1_1171:
АbА
dense_1_1173:	А"
critic_int_1176:	А
critic_int_1178:"
critic_ext_1181:	А
critic_ext_1183:

actor_1186:	А

actor_1188:
identity

identity_1

identity_2Ивactor/StatefulPartitionedCallв conv2d_3/StatefulPartitionedCallв conv2d_4/StatefulPartitionedCallв conv2d_5/StatefulPartitionedCallв"critic_ext/StatefulPartitionedCallв"critic_int/StatefulPartitionedCallвdense_1/StatefulPartitionedCall∙
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_3_1155conv2d_3_1157*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_conv2d_3_layer_call_and_return_conditional_losses_727Ы
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv2d_4_1160conv2d_4_1162*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_conv2d_4_layer_call_and_return_conditional_losses_766Ы
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0conv2d_5_1165conv2d_5_1167*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_conv2d_5_layer_call_and_return_conditional_losses_805▐
flatten_1/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Аb* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_flatten_1_layer_call_and_return_conditional_losses_817Е
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_1171dense_1_1173*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_830Ц
"critic_int/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0critic_int_1176critic_int_1178*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_critic_int_layer_call_and_return_conditional_losses_846Ц
"critic_ext/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0critic_ext_1181critic_ext_1183*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_critic_ext_layer_call_and_return_conditional_losses_862В
actor/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0
actor_1186
actor_1188*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *G
fBR@
>__inference_actor_layer_call_and_return_conditional_losses_879u
IdentityIdentity&actor/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         |

Identity_1Identity+critic_ext/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         |

Identity_2Identity+critic_int/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ╗
NoOpNoOp^actor/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall#^critic_ext/StatefulPartitionedCall#^critic_int/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         TT: : : : : : : : : : : : : : 2>
actor/StatefulPartitionedCallactor/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2H
"critic_ext/StatefulPartitionedCall"critic_ext/StatefulPartitionedCall2H
"critic_int/StatefulPartitionedCall"critic_int/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:\ X
3
_output_shapes!
:         TT
!
_user_specified_name	input_2
пЪ
│
@__inference_policy_layer_call_and_return_conditional_losses_1470

inputsH
.conv2d_3_conv2d_conv2d_readvariableop_resource: I
;conv2d_3_squeeze_batch_dims_biasadd_readvariableop_resource: H
.conv2d_4_conv2d_conv2d_readvariableop_resource: @I
;conv2d_4_squeeze_batch_dims_biasadd_readvariableop_resource:@H
.conv2d_5_conv2d_conv2d_readvariableop_resource:@@I
;conv2d_5_squeeze_batch_dims_biasadd_readvariableop_resource:@:
&dense_1_matmul_readvariableop_resource:
АbА6
'dense_1_biasadd_readvariableop_resource:	А<
)critic_int_matmul_readvariableop_resource:	А8
*critic_int_biasadd_readvariableop_resource:<
)critic_ext_matmul_readvariableop_resource:	А8
*critic_ext_biasadd_readvariableop_resource:7
$actor_matmul_readvariableop_resource:	А3
%actor_biasadd_readvariableop_resource:
identity

identity_1

identity_2Ивactor/BiasAdd/ReadVariableOpвactor/MatMul/ReadVariableOpв%conv2d_3/Conv2D/Conv2D/ReadVariableOpв2conv2d_3/squeeze_batch_dims/BiasAdd/ReadVariableOpв%conv2d_4/Conv2D/Conv2D/ReadVariableOpв2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOpв%conv2d_5/Conv2D/Conv2D/ReadVariableOpв2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOpв!critic_ext/BiasAdd/ReadVariableOpв critic_ext/MatMul/ReadVariableOpв!critic_int/BiasAdd/ReadVariableOpв critic_int/MatMul/ReadVariableOpвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpK
conv2d_3/Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:m
#conv2d_3/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_3/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
¤        o
%conv2d_3/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
conv2d_3/Conv2D/strided_sliceStridedSliceconv2d_3/Conv2D/Shape:output:0,conv2d_3/Conv2D/strided_slice/stack:output:0.conv2d_3/Conv2D/strided_slice/stack_1:output:0.conv2d_3/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_3/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    T   T      М
conv2d_3/Conv2D/ReshapeReshapeinputs&conv2d_3/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:         TTЬ
%conv2d_3/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_3_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0╘
conv2d_3/Conv2D/Conv2DConv2D conv2d_3/Conv2D/Reshape:output:0-conv2d_3/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingVALID*
strides
t
conv2d_3/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"          f
conv2d_3/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         ╚
conv2d_3/Conv2D/concatConcatV2&conv2d_3/Conv2D/strided_slice:output:0(conv2d_3/Conv2D/concat/values_1:output:0$conv2d_3/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:д
conv2d_3/Conv2D/Reshape_1Reshapeconv2d_3/Conv2D/Conv2D:output:0conv2d_3/Conv2D/concat:output:0*
T0*3
_output_shapes!
:          s
!conv2d_3/squeeze_batch_dims/ShapeShape"conv2d_3/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_3/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Д
1conv2d_3/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
¤        {
1conv2d_3/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
)conv2d_3/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_3/squeeze_batch_dims/Shape:output:08conv2d_3/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_3/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_3/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskВ
)conv2d_3/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"              └
#conv2d_3/squeeze_batch_dims/ReshapeReshape"conv2d_3/Conv2D/Reshape_1:output:02conv2d_3/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:          к
2conv2d_3/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_3_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0╥
#conv2d_3/squeeze_batch_dims/BiasAddBiasAdd,conv2d_3/squeeze_batch_dims/Reshape:output:0:conv2d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          А
+conv2d_3/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"          r
'conv2d_3/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         °
"conv2d_3/squeeze_batch_dims/concatConcatV22conv2d_3/squeeze_batch_dims/strided_slice:output:04conv2d_3/squeeze_batch_dims/concat/values_1:output:00conv2d_3/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:╔
%conv2d_3/squeeze_batch_dims/Reshape_1Reshape,conv2d_3/squeeze_batch_dims/BiasAdd:output:0+conv2d_3/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:          Г
conv2d_3/ReluRelu.conv2d_3/squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:          `
conv2d_4/Conv2D/ShapeShapeconv2d_3/Relu:activations:0*
T0*
_output_shapes
:m
#conv2d_4/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_4/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
¤        o
%conv2d_4/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
conv2d_4/Conv2D/strided_sliceStridedSliceconv2d_4/Conv2D/Shape:output:0,conv2d_4/Conv2D/strided_slice/stack:output:0.conv2d_4/Conv2D/strided_slice/stack_1:output:0.conv2d_4/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_4/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"              б
conv2d_4/Conv2D/ReshapeReshapeconv2d_3/Relu:activations:0&conv2d_4/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:          Ь
%conv2d_4/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_4_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0╘
conv2d_4/Conv2D/Conv2DConv2D conv2d_4/Conv2D/Reshape:output:0-conv2d_4/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         		@*
paddingVALID*
strides
t
conv2d_4/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"	   	   @   f
conv2d_4/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         ╚
conv2d_4/Conv2D/concatConcatV2&conv2d_4/Conv2D/strided_slice:output:0(conv2d_4/Conv2D/concat/values_1:output:0$conv2d_4/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:д
conv2d_4/Conv2D/Reshape_1Reshapeconv2d_4/Conv2D/Conv2D:output:0conv2d_4/Conv2D/concat:output:0*
T0*3
_output_shapes!
:         		@s
!conv2d_4/squeeze_batch_dims/ShapeShape"conv2d_4/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_4/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Д
1conv2d_4/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
¤        {
1conv2d_4/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
)conv2d_4/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_4/squeeze_batch_dims/Shape:output:08conv2d_4/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_4/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_4/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskВ
)conv2d_4/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    	   	   @   └
#conv2d_4/squeeze_batch_dims/ReshapeReshape"conv2d_4/Conv2D/Reshape_1:output:02conv2d_4/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:         		@к
2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_4_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╥
#conv2d_4/squeeze_batch_dims/BiasAddBiasAdd,conv2d_4/squeeze_batch_dims/Reshape:output:0:conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         		@А
+conv2d_4/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"	   	   @   r
'conv2d_4/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         °
"conv2d_4/squeeze_batch_dims/concatConcatV22conv2d_4/squeeze_batch_dims/strided_slice:output:04conv2d_4/squeeze_batch_dims/concat/values_1:output:00conv2d_4/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:╔
%conv2d_4/squeeze_batch_dims/Reshape_1Reshape,conv2d_4/squeeze_batch_dims/BiasAdd:output:0+conv2d_4/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:         		@Г
conv2d_4/ReluRelu.conv2d_4/squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:         		@`
conv2d_5/Conv2D/ShapeShapeconv2d_4/Relu:activations:0*
T0*
_output_shapes
:m
#conv2d_5/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_5/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
¤        o
%conv2d_5/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
conv2d_5/Conv2D/strided_sliceStridedSliceconv2d_5/Conv2D/Shape:output:0,conv2d_5/Conv2D/strided_slice/stack:output:0.conv2d_5/Conv2D/strided_slice/stack_1:output:0.conv2d_5/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_5/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    	   	   @   б
conv2d_5/Conv2D/ReshapeReshapeconv2d_4/Relu:activations:0&conv2d_5/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:         		@Ь
%conv2d_5/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_5_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0╘
conv2d_5/Conv2D/Conv2DConv2D conv2d_5/Conv2D/Reshape:output:0-conv2d_5/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
t
conv2d_5/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      @   f
conv2d_5/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         ╚
conv2d_5/Conv2D/concatConcatV2&conv2d_5/Conv2D/strided_slice:output:0(conv2d_5/Conv2D/concat/values_1:output:0$conv2d_5/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:д
conv2d_5/Conv2D/Reshape_1Reshapeconv2d_5/Conv2D/Conv2D:output:0conv2d_5/Conv2D/concat:output:0*
T0*3
_output_shapes!
:         @s
!conv2d_5/squeeze_batch_dims/ShapeShape"conv2d_5/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_5/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Д
1conv2d_5/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
¤        {
1conv2d_5/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
)conv2d_5/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_5/squeeze_batch_dims/Shape:output:08conv2d_5/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_5/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_5/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskВ
)conv2d_5/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"          @   └
#conv2d_5/squeeze_batch_dims/ReshapeReshape"conv2d_5/Conv2D/Reshape_1:output:02conv2d_5/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:         @к
2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_5_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╥
#conv2d_5/squeeze_batch_dims/BiasAddBiasAdd,conv2d_5/squeeze_batch_dims/Reshape:output:0:conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @А
+conv2d_5/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      @   r
'conv2d_5/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         °
"conv2d_5/squeeze_batch_dims/concatConcatV22conv2d_5/squeeze_batch_dims/strided_slice:output:04conv2d_5/squeeze_batch_dims/concat/values_1:output:00conv2d_5/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:╔
%conv2d_5/squeeze_batch_dims/Reshape_1Reshape,conv2d_5/squeeze_batch_dims/BiasAdd:output:0+conv2d_5/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:         @Г
conv2d_5/ReluRelu.conv2d_5/squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:         @`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"     1  Ж
flatten_1/ReshapeReshapeconv2d_5/Relu:activations:0flatten_1/Const:output:0*
T0*(
_output_shapes
:         АbЖ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
АbА*
dtype0О
dense_1/MatMulMatMulflatten_1/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АГ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0П
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аa
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:         АЛ
 critic_int/MatMul/ReadVariableOpReadVariableOp)critic_int_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0У
critic_int/MatMulMatMuldense_1/Relu:activations:0(critic_int/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         И
!critic_int/BiasAdd/ReadVariableOpReadVariableOp*critic_int_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ч
critic_int/BiasAddBiasAddcritic_int/MatMul:product:0)critic_int/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Л
 critic_ext/MatMul/ReadVariableOpReadVariableOp)critic_ext_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0У
critic_ext/MatMulMatMuldense_1/Relu:activations:0(critic_ext/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         И
!critic_ext/BiasAdd/ReadVariableOpReadVariableOp*critic_ext_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ч
critic_ext/BiasAddBiasAddcritic_ext/MatMul:product:0)critic_ext/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Б
actor/MatMul/ReadVariableOpReadVariableOp$actor_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Й
actor/MatMulMatMuldense_1/Relu:activations:0#actor/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ~
actor/BiasAdd/ReadVariableOpReadVariableOp%actor_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0И
actor/BiasAddBiasAddactor/MatMul:product:0$actor/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         b
actor/SoftmaxSoftmaxactor/BiasAdd:output:0*
T0*'
_output_shapes
:         f
IdentityIdentityactor/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         l

Identity_1Identitycritic_ext/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         l

Identity_2Identitycritic_int/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         щ
NoOpNoOp^actor/BiasAdd/ReadVariableOp^actor/MatMul/ReadVariableOp&^conv2d_3/Conv2D/Conv2D/ReadVariableOp3^conv2d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_4/Conv2D/Conv2D/ReadVariableOp3^conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_5/Conv2D/Conv2D/ReadVariableOp3^conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp"^critic_ext/BiasAdd/ReadVariableOp!^critic_ext/MatMul/ReadVariableOp"^critic_int/BiasAdd/ReadVariableOp!^critic_int/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         TT: : : : : : : : : : : : : : 2<
actor/BiasAdd/ReadVariableOpactor/BiasAdd/ReadVariableOp2:
actor/MatMul/ReadVariableOpactor/MatMul/ReadVariableOp2N
%conv2d_3/Conv2D/Conv2D/ReadVariableOp%conv2d_3/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_4/Conv2D/Conv2D/ReadVariableOp%conv2d_4/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_5/Conv2D/Conv2D/ReadVariableOp%conv2d_5/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp2F
!critic_ext/BiasAdd/ReadVariableOp!critic_ext/BiasAdd/ReadVariableOp2D
 critic_ext/MatMul/ReadVariableOp critic_ext/MatMul/ReadVariableOp2F
!critic_int/BiasAdd/ReadVariableOp!critic_int/BiasAdd/ReadVariableOp2D
 critic_int/MatMul/ReadVariableOp critic_int/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:[ W
3
_output_shapes!
:         TT
 
_user_specified_nameinputs
пЪ
│
@__inference_policy_layer_call_and_return_conditional_losses_1591

inputsH
.conv2d_3_conv2d_conv2d_readvariableop_resource: I
;conv2d_3_squeeze_batch_dims_biasadd_readvariableop_resource: H
.conv2d_4_conv2d_conv2d_readvariableop_resource: @I
;conv2d_4_squeeze_batch_dims_biasadd_readvariableop_resource:@H
.conv2d_5_conv2d_conv2d_readvariableop_resource:@@I
;conv2d_5_squeeze_batch_dims_biasadd_readvariableop_resource:@:
&dense_1_matmul_readvariableop_resource:
АbА6
'dense_1_biasadd_readvariableop_resource:	А<
)critic_int_matmul_readvariableop_resource:	А8
*critic_int_biasadd_readvariableop_resource:<
)critic_ext_matmul_readvariableop_resource:	А8
*critic_ext_biasadd_readvariableop_resource:7
$actor_matmul_readvariableop_resource:	А3
%actor_biasadd_readvariableop_resource:
identity

identity_1

identity_2Ивactor/BiasAdd/ReadVariableOpвactor/MatMul/ReadVariableOpв%conv2d_3/Conv2D/Conv2D/ReadVariableOpв2conv2d_3/squeeze_batch_dims/BiasAdd/ReadVariableOpв%conv2d_4/Conv2D/Conv2D/ReadVariableOpв2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOpв%conv2d_5/Conv2D/Conv2D/ReadVariableOpв2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOpв!critic_ext/BiasAdd/ReadVariableOpв critic_ext/MatMul/ReadVariableOpв!critic_int/BiasAdd/ReadVariableOpв critic_int/MatMul/ReadVariableOpвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpK
conv2d_3/Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:m
#conv2d_3/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_3/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
¤        o
%conv2d_3/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
conv2d_3/Conv2D/strided_sliceStridedSliceconv2d_3/Conv2D/Shape:output:0,conv2d_3/Conv2D/strided_slice/stack:output:0.conv2d_3/Conv2D/strided_slice/stack_1:output:0.conv2d_3/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_3/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    T   T      М
conv2d_3/Conv2D/ReshapeReshapeinputs&conv2d_3/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:         TTЬ
%conv2d_3/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_3_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0╘
conv2d_3/Conv2D/Conv2DConv2D conv2d_3/Conv2D/Reshape:output:0-conv2d_3/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingVALID*
strides
t
conv2d_3/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"          f
conv2d_3/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         ╚
conv2d_3/Conv2D/concatConcatV2&conv2d_3/Conv2D/strided_slice:output:0(conv2d_3/Conv2D/concat/values_1:output:0$conv2d_3/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:д
conv2d_3/Conv2D/Reshape_1Reshapeconv2d_3/Conv2D/Conv2D:output:0conv2d_3/Conv2D/concat:output:0*
T0*3
_output_shapes!
:          s
!conv2d_3/squeeze_batch_dims/ShapeShape"conv2d_3/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_3/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Д
1conv2d_3/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
¤        {
1conv2d_3/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
)conv2d_3/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_3/squeeze_batch_dims/Shape:output:08conv2d_3/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_3/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_3/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskВ
)conv2d_3/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"              └
#conv2d_3/squeeze_batch_dims/ReshapeReshape"conv2d_3/Conv2D/Reshape_1:output:02conv2d_3/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:          к
2conv2d_3/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_3_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0╥
#conv2d_3/squeeze_batch_dims/BiasAddBiasAdd,conv2d_3/squeeze_batch_dims/Reshape:output:0:conv2d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          А
+conv2d_3/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"          r
'conv2d_3/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         °
"conv2d_3/squeeze_batch_dims/concatConcatV22conv2d_3/squeeze_batch_dims/strided_slice:output:04conv2d_3/squeeze_batch_dims/concat/values_1:output:00conv2d_3/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:╔
%conv2d_3/squeeze_batch_dims/Reshape_1Reshape,conv2d_3/squeeze_batch_dims/BiasAdd:output:0+conv2d_3/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:          Г
conv2d_3/ReluRelu.conv2d_3/squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:          `
conv2d_4/Conv2D/ShapeShapeconv2d_3/Relu:activations:0*
T0*
_output_shapes
:m
#conv2d_4/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_4/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
¤        o
%conv2d_4/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
conv2d_4/Conv2D/strided_sliceStridedSliceconv2d_4/Conv2D/Shape:output:0,conv2d_4/Conv2D/strided_slice/stack:output:0.conv2d_4/Conv2D/strided_slice/stack_1:output:0.conv2d_4/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_4/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"              б
conv2d_4/Conv2D/ReshapeReshapeconv2d_3/Relu:activations:0&conv2d_4/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:          Ь
%conv2d_4/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_4_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0╘
conv2d_4/Conv2D/Conv2DConv2D conv2d_4/Conv2D/Reshape:output:0-conv2d_4/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         		@*
paddingVALID*
strides
t
conv2d_4/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"	   	   @   f
conv2d_4/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         ╚
conv2d_4/Conv2D/concatConcatV2&conv2d_4/Conv2D/strided_slice:output:0(conv2d_4/Conv2D/concat/values_1:output:0$conv2d_4/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:д
conv2d_4/Conv2D/Reshape_1Reshapeconv2d_4/Conv2D/Conv2D:output:0conv2d_4/Conv2D/concat:output:0*
T0*3
_output_shapes!
:         		@s
!conv2d_4/squeeze_batch_dims/ShapeShape"conv2d_4/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_4/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Д
1conv2d_4/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
¤        {
1conv2d_4/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
)conv2d_4/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_4/squeeze_batch_dims/Shape:output:08conv2d_4/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_4/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_4/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskВ
)conv2d_4/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    	   	   @   └
#conv2d_4/squeeze_batch_dims/ReshapeReshape"conv2d_4/Conv2D/Reshape_1:output:02conv2d_4/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:         		@к
2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_4_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╥
#conv2d_4/squeeze_batch_dims/BiasAddBiasAdd,conv2d_4/squeeze_batch_dims/Reshape:output:0:conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         		@А
+conv2d_4/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"	   	   @   r
'conv2d_4/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         °
"conv2d_4/squeeze_batch_dims/concatConcatV22conv2d_4/squeeze_batch_dims/strided_slice:output:04conv2d_4/squeeze_batch_dims/concat/values_1:output:00conv2d_4/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:╔
%conv2d_4/squeeze_batch_dims/Reshape_1Reshape,conv2d_4/squeeze_batch_dims/BiasAdd:output:0+conv2d_4/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:         		@Г
conv2d_4/ReluRelu.conv2d_4/squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:         		@`
conv2d_5/Conv2D/ShapeShapeconv2d_4/Relu:activations:0*
T0*
_output_shapes
:m
#conv2d_5/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_5/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
¤        o
%conv2d_5/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
conv2d_5/Conv2D/strided_sliceStridedSliceconv2d_5/Conv2D/Shape:output:0,conv2d_5/Conv2D/strided_slice/stack:output:0.conv2d_5/Conv2D/strided_slice/stack_1:output:0.conv2d_5/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_5/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    	   	   @   б
conv2d_5/Conv2D/ReshapeReshapeconv2d_4/Relu:activations:0&conv2d_5/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:         		@Ь
%conv2d_5/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_5_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0╘
conv2d_5/Conv2D/Conv2DConv2D conv2d_5/Conv2D/Reshape:output:0-conv2d_5/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
t
conv2d_5/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      @   f
conv2d_5/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         ╚
conv2d_5/Conv2D/concatConcatV2&conv2d_5/Conv2D/strided_slice:output:0(conv2d_5/Conv2D/concat/values_1:output:0$conv2d_5/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:д
conv2d_5/Conv2D/Reshape_1Reshapeconv2d_5/Conv2D/Conv2D:output:0conv2d_5/Conv2D/concat:output:0*
T0*3
_output_shapes!
:         @s
!conv2d_5/squeeze_batch_dims/ShapeShape"conv2d_5/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_5/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Д
1conv2d_5/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
¤        {
1conv2d_5/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
)conv2d_5/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_5/squeeze_batch_dims/Shape:output:08conv2d_5/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_5/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_5/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskВ
)conv2d_5/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"          @   └
#conv2d_5/squeeze_batch_dims/ReshapeReshape"conv2d_5/Conv2D/Reshape_1:output:02conv2d_5/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:         @к
2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_5_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╥
#conv2d_5/squeeze_batch_dims/BiasAddBiasAdd,conv2d_5/squeeze_batch_dims/Reshape:output:0:conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @А
+conv2d_5/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      @   r
'conv2d_5/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         °
"conv2d_5/squeeze_batch_dims/concatConcatV22conv2d_5/squeeze_batch_dims/strided_slice:output:04conv2d_5/squeeze_batch_dims/concat/values_1:output:00conv2d_5/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:╔
%conv2d_5/squeeze_batch_dims/Reshape_1Reshape,conv2d_5/squeeze_batch_dims/BiasAdd:output:0+conv2d_5/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:         @Г
conv2d_5/ReluRelu.conv2d_5/squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:         @`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"     1  Ж
flatten_1/ReshapeReshapeconv2d_5/Relu:activations:0flatten_1/Const:output:0*
T0*(
_output_shapes
:         АbЖ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
АbА*
dtype0О
dense_1/MatMulMatMulflatten_1/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АГ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0П
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аa
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:         АЛ
 critic_int/MatMul/ReadVariableOpReadVariableOp)critic_int_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0У
critic_int/MatMulMatMuldense_1/Relu:activations:0(critic_int/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         И
!critic_int/BiasAdd/ReadVariableOpReadVariableOp*critic_int_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ч
critic_int/BiasAddBiasAddcritic_int/MatMul:product:0)critic_int/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Л
 critic_ext/MatMul/ReadVariableOpReadVariableOp)critic_ext_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0У
critic_ext/MatMulMatMuldense_1/Relu:activations:0(critic_ext/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         И
!critic_ext/BiasAdd/ReadVariableOpReadVariableOp*critic_ext_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ч
critic_ext/BiasAddBiasAddcritic_ext/MatMul:product:0)critic_ext/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Б
actor/MatMul/ReadVariableOpReadVariableOp$actor_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Й
actor/MatMulMatMuldense_1/Relu:activations:0#actor/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ~
actor/BiasAdd/ReadVariableOpReadVariableOp%actor_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0И
actor/BiasAddBiasAddactor/MatMul:product:0$actor/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         b
actor/SoftmaxSoftmaxactor/BiasAdd:output:0*
T0*'
_output_shapes
:         f
IdentityIdentityactor/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         l

Identity_1Identitycritic_ext/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         l

Identity_2Identitycritic_int/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         щ
NoOpNoOp^actor/BiasAdd/ReadVariableOp^actor/MatMul/ReadVariableOp&^conv2d_3/Conv2D/Conv2D/ReadVariableOp3^conv2d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_4/Conv2D/Conv2D/ReadVariableOp3^conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_5/Conv2D/Conv2D/ReadVariableOp3^conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp"^critic_ext/BiasAdd/ReadVariableOp!^critic_ext/MatMul/ReadVariableOp"^critic_int/BiasAdd/ReadVariableOp!^critic_int/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         TT: : : : : : : : : : : : : : 2<
actor/BiasAdd/ReadVariableOpactor/BiasAdd/ReadVariableOp2:
actor/MatMul/ReadVariableOpactor/MatMul/ReadVariableOp2N
%conv2d_3/Conv2D/Conv2D/ReadVariableOp%conv2d_3/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_4/Conv2D/Conv2D/ReadVariableOp%conv2d_4/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_5/Conv2D/Conv2D/ReadVariableOp%conv2d_5/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp2F
!critic_ext/BiasAdd/ReadVariableOp!critic_ext/BiasAdd/ReadVariableOp2D
 critic_ext/MatMul/ReadVariableOp critic_ext/MatMul/ReadVariableOp2F
!critic_int/BiasAdd/ReadVariableOp!critic_int/BiasAdd/ReadVariableOp2D
 critic_int/MatMul/ReadVariableOp critic_int/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:[ W
3
_output_shapes!
:         TT
 
_user_specified_nameinputs
¤>
┘
 __inference__traced_restore_1925
file_prefix:
 assignvariableop_conv2d_3_kernel: .
 assignvariableop_1_conv2d_3_bias: <
"assignvariableop_2_conv2d_4_kernel: @.
 assignvariableop_3_conv2d_4_bias:@<
"assignvariableop_4_conv2d_5_kernel:@@.
 assignvariableop_5_conv2d_5_bias:@5
!assignvariableop_6_dense_1_kernel:
АbА.
assignvariableop_7_dense_1_bias:	А2
assignvariableop_8_actor_kernel:	А+
assignvariableop_9_actor_bias:8
%assignvariableop_10_critic_ext_kernel:	А1
#assignvariableop_11_critic_ext_bias:8
%assignvariableop_12_critic_int_kernel:	А1
#assignvariableop_13_critic_int_bias:
identity_15ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_2вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9П
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*╡
valueлBиB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHО
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B щ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*P
_output_shapes>
<:::::::::::::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:│
AssignVariableOpAssignVariableOp assignvariableop_conv2d_3_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_3_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_4_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_4_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_5_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_5_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:╕
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_1_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:╢
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_1_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:╢
AssignVariableOp_8AssignVariableOpassignvariableop_8_actor_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_9AssignVariableOpassignvariableop_9_actor_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_10AssignVariableOp%assignvariableop_10_critic_ext_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:╝
AssignVariableOp_11AssignVariableOp#assignvariableop_11_critic_ext_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_12AssignVariableOp%assignvariableop_12_critic_int_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:╝
AssignVariableOp_13AssignVariableOp#assignvariableop_13_critic_int_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 Г
Identity_14Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_15IdentityIdentity_14:output:0^NoOp_1*
T0*
_output_shapes
: Ё
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_15Identity_15:output:0*1
_input_shapes 
: : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Я

ё
?__inference_actor_layer_call_and_return_conditional_losses_1768

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Юк
╓
__inference__wrapped_model_687
input_2O
5policy_conv2d_3_conv2d_conv2d_readvariableop_resource: P
Bpolicy_conv2d_3_squeeze_batch_dims_biasadd_readvariableop_resource: O
5policy_conv2d_4_conv2d_conv2d_readvariableop_resource: @P
Bpolicy_conv2d_4_squeeze_batch_dims_biasadd_readvariableop_resource:@O
5policy_conv2d_5_conv2d_conv2d_readvariableop_resource:@@P
Bpolicy_conv2d_5_squeeze_batch_dims_biasadd_readvariableop_resource:@A
-policy_dense_1_matmul_readvariableop_resource:
АbА=
.policy_dense_1_biasadd_readvariableop_resource:	АC
0policy_critic_int_matmul_readvariableop_resource:	А?
1policy_critic_int_biasadd_readvariableop_resource:C
0policy_critic_ext_matmul_readvariableop_resource:	А?
1policy_critic_ext_biasadd_readvariableop_resource:>
+policy_actor_matmul_readvariableop_resource:	А:
,policy_actor_biasadd_readvariableop_resource:
identity

identity_1

identity_2Ив#policy/actor/BiasAdd/ReadVariableOpв"policy/actor/MatMul/ReadVariableOpв,policy/conv2d_3/Conv2D/Conv2D/ReadVariableOpв9policy/conv2d_3/squeeze_batch_dims/BiasAdd/ReadVariableOpв,policy/conv2d_4/Conv2D/Conv2D/ReadVariableOpв9policy/conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOpв,policy/conv2d_5/Conv2D/Conv2D/ReadVariableOpв9policy/conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOpв(policy/critic_ext/BiasAdd/ReadVariableOpв'policy/critic_ext/MatMul/ReadVariableOpв(policy/critic_int/BiasAdd/ReadVariableOpв'policy/critic_int/MatMul/ReadVariableOpв%policy/dense_1/BiasAdd/ReadVariableOpв$policy/dense_1/MatMul/ReadVariableOpS
policy/conv2d_3/Conv2D/ShapeShapeinput_2*
T0*
_output_shapes
:t
*policy/conv2d_3/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
,policy/conv2d_3/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
¤        v
,policy/conv2d_3/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┬
$policy/conv2d_3/Conv2D/strided_sliceStridedSlice%policy/conv2d_3/Conv2D/Shape:output:03policy/conv2d_3/Conv2D/strided_slice/stack:output:05policy/conv2d_3/Conv2D/strided_slice/stack_1:output:05policy/conv2d_3/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask}
$policy/conv2d_3/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    T   T      Ы
policy/conv2d_3/Conv2D/ReshapeReshapeinput_2-policy/conv2d_3/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:         TTк
,policy/conv2d_3/Conv2D/Conv2D/ReadVariableOpReadVariableOp5policy_conv2d_3_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0щ
policy/conv2d_3/Conv2D/Conv2DConv2D'policy/conv2d_3/Conv2D/Reshape:output:04policy/conv2d_3/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingVALID*
strides
{
&policy/conv2d_3/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"          m
"policy/conv2d_3/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         ф
policy/conv2d_3/Conv2D/concatConcatV2-policy/conv2d_3/Conv2D/strided_slice:output:0/policy/conv2d_3/Conv2D/concat/values_1:output:0+policy/conv2d_3/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:╣
 policy/conv2d_3/Conv2D/Reshape_1Reshape&policy/conv2d_3/Conv2D/Conv2D:output:0&policy/conv2d_3/Conv2D/concat:output:0*
T0*3
_output_shapes!
:          Б
(policy/conv2d_3/squeeze_batch_dims/ShapeShape)policy/conv2d_3/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:А
6policy/conv2d_3/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Л
8policy/conv2d_3/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
¤        В
8policy/conv2d_3/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:■
0policy/conv2d_3/squeeze_batch_dims/strided_sliceStridedSlice1policy/conv2d_3/squeeze_batch_dims/Shape:output:0?policy/conv2d_3/squeeze_batch_dims/strided_slice/stack:output:0Apolicy/conv2d_3/squeeze_batch_dims/strided_slice/stack_1:output:0Apolicy/conv2d_3/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskЙ
0policy/conv2d_3/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"              ╒
*policy/conv2d_3/squeeze_batch_dims/ReshapeReshape)policy/conv2d_3/Conv2D/Reshape_1:output:09policy/conv2d_3/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:          ╕
9policy/conv2d_3/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOpBpolicy_conv2d_3_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ч
*policy/conv2d_3/squeeze_batch_dims/BiasAddBiasAdd3policy/conv2d_3/squeeze_batch_dims/Reshape:output:0Apolicy/conv2d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          З
2policy/conv2d_3/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"          y
.policy/conv2d_3/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         Ф
)policy/conv2d_3/squeeze_batch_dims/concatConcatV29policy/conv2d_3/squeeze_batch_dims/strided_slice:output:0;policy/conv2d_3/squeeze_batch_dims/concat/values_1:output:07policy/conv2d_3/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:▐
,policy/conv2d_3/squeeze_batch_dims/Reshape_1Reshape3policy/conv2d_3/squeeze_batch_dims/BiasAdd:output:02policy/conv2d_3/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:          С
policy/conv2d_3/ReluRelu5policy/conv2d_3/squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:          n
policy/conv2d_4/Conv2D/ShapeShape"policy/conv2d_3/Relu:activations:0*
T0*
_output_shapes
:t
*policy/conv2d_4/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
,policy/conv2d_4/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
¤        v
,policy/conv2d_4/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┬
$policy/conv2d_4/Conv2D/strided_sliceStridedSlice%policy/conv2d_4/Conv2D/Shape:output:03policy/conv2d_4/Conv2D/strided_slice/stack:output:05policy/conv2d_4/Conv2D/strided_slice/stack_1:output:05policy/conv2d_4/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask}
$policy/conv2d_4/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"              ╢
policy/conv2d_4/Conv2D/ReshapeReshape"policy/conv2d_3/Relu:activations:0-policy/conv2d_4/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:          к
,policy/conv2d_4/Conv2D/Conv2D/ReadVariableOpReadVariableOp5policy_conv2d_4_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0щ
policy/conv2d_4/Conv2D/Conv2DConv2D'policy/conv2d_4/Conv2D/Reshape:output:04policy/conv2d_4/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         		@*
paddingVALID*
strides
{
&policy/conv2d_4/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"	   	   @   m
"policy/conv2d_4/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         ф
policy/conv2d_4/Conv2D/concatConcatV2-policy/conv2d_4/Conv2D/strided_slice:output:0/policy/conv2d_4/Conv2D/concat/values_1:output:0+policy/conv2d_4/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:╣
 policy/conv2d_4/Conv2D/Reshape_1Reshape&policy/conv2d_4/Conv2D/Conv2D:output:0&policy/conv2d_4/Conv2D/concat:output:0*
T0*3
_output_shapes!
:         		@Б
(policy/conv2d_4/squeeze_batch_dims/ShapeShape)policy/conv2d_4/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:А
6policy/conv2d_4/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Л
8policy/conv2d_4/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
¤        В
8policy/conv2d_4/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:■
0policy/conv2d_4/squeeze_batch_dims/strided_sliceStridedSlice1policy/conv2d_4/squeeze_batch_dims/Shape:output:0?policy/conv2d_4/squeeze_batch_dims/strided_slice/stack:output:0Apolicy/conv2d_4/squeeze_batch_dims/strided_slice/stack_1:output:0Apolicy/conv2d_4/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskЙ
0policy/conv2d_4/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    	   	   @   ╒
*policy/conv2d_4/squeeze_batch_dims/ReshapeReshape)policy/conv2d_4/Conv2D/Reshape_1:output:09policy/conv2d_4/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:         		@╕
9policy/conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOpBpolicy_conv2d_4_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ч
*policy/conv2d_4/squeeze_batch_dims/BiasAddBiasAdd3policy/conv2d_4/squeeze_batch_dims/Reshape:output:0Apolicy/conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         		@З
2policy/conv2d_4/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"	   	   @   y
.policy/conv2d_4/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         Ф
)policy/conv2d_4/squeeze_batch_dims/concatConcatV29policy/conv2d_4/squeeze_batch_dims/strided_slice:output:0;policy/conv2d_4/squeeze_batch_dims/concat/values_1:output:07policy/conv2d_4/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:▐
,policy/conv2d_4/squeeze_batch_dims/Reshape_1Reshape3policy/conv2d_4/squeeze_batch_dims/BiasAdd:output:02policy/conv2d_4/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:         		@С
policy/conv2d_4/ReluRelu5policy/conv2d_4/squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:         		@n
policy/conv2d_5/Conv2D/ShapeShape"policy/conv2d_4/Relu:activations:0*
T0*
_output_shapes
:t
*policy/conv2d_5/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
,policy/conv2d_5/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
¤        v
,policy/conv2d_5/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┬
$policy/conv2d_5/Conv2D/strided_sliceStridedSlice%policy/conv2d_5/Conv2D/Shape:output:03policy/conv2d_5/Conv2D/strided_slice/stack:output:05policy/conv2d_5/Conv2D/strided_slice/stack_1:output:05policy/conv2d_5/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask}
$policy/conv2d_5/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"    	   	   @   ╢
policy/conv2d_5/Conv2D/ReshapeReshape"policy/conv2d_4/Relu:activations:0-policy/conv2d_5/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:         		@к
,policy/conv2d_5/Conv2D/Conv2D/ReadVariableOpReadVariableOp5policy_conv2d_5_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0щ
policy/conv2d_5/Conv2D/Conv2DConv2D'policy/conv2d_5/Conv2D/Reshape:output:04policy/conv2d_5/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
{
&policy/conv2d_5/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      @   m
"policy/conv2d_5/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         ф
policy/conv2d_5/Conv2D/concatConcatV2-policy/conv2d_5/Conv2D/strided_slice:output:0/policy/conv2d_5/Conv2D/concat/values_1:output:0+policy/conv2d_5/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:╣
 policy/conv2d_5/Conv2D/Reshape_1Reshape&policy/conv2d_5/Conv2D/Conv2D:output:0&policy/conv2d_5/Conv2D/concat:output:0*
T0*3
_output_shapes!
:         @Б
(policy/conv2d_5/squeeze_batch_dims/ShapeShape)policy/conv2d_5/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:А
6policy/conv2d_5/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Л
8policy/conv2d_5/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
¤        В
8policy/conv2d_5/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:■
0policy/conv2d_5/squeeze_batch_dims/strided_sliceStridedSlice1policy/conv2d_5/squeeze_batch_dims/Shape:output:0?policy/conv2d_5/squeeze_batch_dims/strided_slice/stack:output:0Apolicy/conv2d_5/squeeze_batch_dims/strided_slice/stack_1:output:0Apolicy/conv2d_5/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskЙ
0policy/conv2d_5/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"          @   ╒
*policy/conv2d_5/squeeze_batch_dims/ReshapeReshape)policy/conv2d_5/Conv2D/Reshape_1:output:09policy/conv2d_5/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:         @╕
9policy/conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOpBpolicy_conv2d_5_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ч
*policy/conv2d_5/squeeze_batch_dims/BiasAddBiasAdd3policy/conv2d_5/squeeze_batch_dims/Reshape:output:0Apolicy/conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @З
2policy/conv2d_5/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      @   y
.policy/conv2d_5/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         Ф
)policy/conv2d_5/squeeze_batch_dims/concatConcatV29policy/conv2d_5/squeeze_batch_dims/strided_slice:output:0;policy/conv2d_5/squeeze_batch_dims/concat/values_1:output:07policy/conv2d_5/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:▐
,policy/conv2d_5/squeeze_batch_dims/Reshape_1Reshape3policy/conv2d_5/squeeze_batch_dims/BiasAdd:output:02policy/conv2d_5/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:         @С
policy/conv2d_5/ReluRelu5policy/conv2d_5/squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:         @g
policy/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"     1  Ы
policy/flatten_1/ReshapeReshape"policy/conv2d_5/Relu:activations:0policy/flatten_1/Const:output:0*
T0*(
_output_shapes
:         АbФ
$policy/dense_1/MatMul/ReadVariableOpReadVariableOp-policy_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
АbА*
dtype0г
policy/dense_1/MatMulMatMul!policy/flatten_1/Reshape:output:0,policy/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АС
%policy/dense_1/BiasAdd/ReadVariableOpReadVariableOp.policy_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0д
policy/dense_1/BiasAddBiasAddpolicy/dense_1/MatMul:product:0-policy/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аo
policy/dense_1/ReluRelupolicy/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:         АЩ
'policy/critic_int/MatMul/ReadVariableOpReadVariableOp0policy_critic_int_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0и
policy/critic_int/MatMulMatMul!policy/dense_1/Relu:activations:0/policy/critic_int/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ц
(policy/critic_int/BiasAdd/ReadVariableOpReadVariableOp1policy_critic_int_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0м
policy/critic_int/BiasAddBiasAdd"policy/critic_int/MatMul:product:00policy/critic_int/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Щ
'policy/critic_ext/MatMul/ReadVariableOpReadVariableOp0policy_critic_ext_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0и
policy/critic_ext/MatMulMatMul!policy/dense_1/Relu:activations:0/policy/critic_ext/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ц
(policy/critic_ext/BiasAdd/ReadVariableOpReadVariableOp1policy_critic_ext_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0м
policy/critic_ext/BiasAddBiasAdd"policy/critic_ext/MatMul:product:00policy/critic_ext/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         П
"policy/actor/MatMul/ReadVariableOpReadVariableOp+policy_actor_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Ю
policy/actor/MatMulMatMul!policy/dense_1/Relu:activations:0*policy/actor/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         М
#policy/actor/BiasAdd/ReadVariableOpReadVariableOp,policy_actor_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Э
policy/actor/BiasAddBiasAddpolicy/actor/MatMul:product:0+policy/actor/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         p
policy/actor/SoftmaxSoftmaxpolicy/actor/BiasAdd:output:0*
T0*'
_output_shapes
:         m
IdentityIdentitypolicy/actor/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         s

Identity_1Identity"policy/critic_ext/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         s

Identity_2Identity"policy/critic_int/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         ╦
NoOpNoOp$^policy/actor/BiasAdd/ReadVariableOp#^policy/actor/MatMul/ReadVariableOp-^policy/conv2d_3/Conv2D/Conv2D/ReadVariableOp:^policy/conv2d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp-^policy/conv2d_4/Conv2D/Conv2D/ReadVariableOp:^policy/conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp-^policy/conv2d_5/Conv2D/Conv2D/ReadVariableOp:^policy/conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp)^policy/critic_ext/BiasAdd/ReadVariableOp(^policy/critic_ext/MatMul/ReadVariableOp)^policy/critic_int/BiasAdd/ReadVariableOp(^policy/critic_int/MatMul/ReadVariableOp&^policy/dense_1/BiasAdd/ReadVariableOp%^policy/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         TT: : : : : : : : : : : : : : 2J
#policy/actor/BiasAdd/ReadVariableOp#policy/actor/BiasAdd/ReadVariableOp2H
"policy/actor/MatMul/ReadVariableOp"policy/actor/MatMul/ReadVariableOp2\
,policy/conv2d_3/Conv2D/Conv2D/ReadVariableOp,policy/conv2d_3/Conv2D/Conv2D/ReadVariableOp2v
9policy/conv2d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp9policy/conv2d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp2\
,policy/conv2d_4/Conv2D/Conv2D/ReadVariableOp,policy/conv2d_4/Conv2D/Conv2D/ReadVariableOp2v
9policy/conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp9policy/conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp2\
,policy/conv2d_5/Conv2D/Conv2D/ReadVariableOp,policy/conv2d_5/Conv2D/Conv2D/ReadVariableOp2v
9policy/conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp9policy/conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp2T
(policy/critic_ext/BiasAdd/ReadVariableOp(policy/critic_ext/BiasAdd/ReadVariableOp2R
'policy/critic_ext/MatMul/ReadVariableOp'policy/critic_ext/MatMul/ReadVariableOp2T
(policy/critic_int/BiasAdd/ReadVariableOp(policy/critic_int/BiasAdd/ReadVariableOp2R
'policy/critic_int/MatMul/ReadVariableOp'policy/critic_int/MatMul/ReadVariableOp2N
%policy/dense_1/BiasAdd/ReadVariableOp%policy/dense_1/BiasAdd/ReadVariableOp2L
$policy/dense_1/MatMul/ReadVariableOp$policy/dense_1/MatMul/ReadVariableOp:\ X
3
_output_shapes!
:         TT
!
_user_specified_name	input_2
°
Ь
'__inference_conv2d_4_layer_call_fn_1642

inputs!
unknown: @
	unknown_0:@
identityИвStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_conv2d_4_layer_call_and_return_conditional_losses_766{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:         		@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:          
 
_user_specified_nameinputs
╖
D
(__inference_flatten_1_layer_call_fn_1722

inputs
identity▒
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Аb* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_flatten_1_layer_call_and_return_conditional_losses_817a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         Аb"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @:[ W
3
_output_shapes!
:         @
 
_user_specified_nameinputs
х
Ч
%__inference_policy_layer_call_fn_1349

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:
АbА
	unknown_6:	А
	unknown_7:	А
	unknown_8:
	unknown_9:	А

unknown_10:

unknown_11:	А

unknown_12:
identity

identity_1

identity_2ИвStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_policy_layer_call_and_return_conditional_losses_1080o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         TT: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:         TT
 
_user_specified_nameinputs
╟
Ч
)__inference_critic_ext_layer_call_fn_1777

inputs
unknown:	А
	unknown_0:
identityИвStatefulPartitionedCall█
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_critic_ext_layer_call_and_return_conditional_losses_862o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╩	
ї
C__inference_critic_int_layer_call_and_return_conditional_losses_846

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs"Ж
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*┤
serving_defaultа
G
input_2<
serving_default_input_2:0         TT9
actor0
StatefulPartitionedCall:0         >

critic_ext0
StatefulPartitionedCall:1         >

critic_int0
StatefulPartitionedCall:2         tensorflow/serving/predict:╫╟
 
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
▌
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op"
_tf_keras_layer
▌
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses

!kernel
"bias
 #_jit_compiled_convolution_op"
_tf_keras_layer
▌
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses

*kernel
+bias
 ,_jit_compiled_convolution_op"
_tf_keras_layer
е
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses"
_tf_keras_layer
╗
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses

9kernel
:bias"
_tf_keras_layer
╗
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses

Akernel
Bbias"
_tf_keras_layer
╗
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses

Ikernel
Jbias"
_tf_keras_layer
╗
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses

Qkernel
Rbias"
_tf_keras_layer
Ж
0
1
!2
"3
*4
+5
96
:7
A8
B9
I10
J11
Q12
R13"
trackable_list_wrapper
Ж
0
1
!2
"3
*4
+5
96
:7
A8
B9
I10
J11
Q12
R13"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
╚
Xtrace_0
Ytrace_1
Ztrace_2
[trace_32▌
$__inference_policy_layer_call_fn_923
%__inference_policy_layer_call_fn_1312
%__inference_policy_layer_call_fn_1349
%__inference_policy_layer_call_fn_1152┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zXtrace_0zYtrace_1zZtrace_2z[trace_3
╡
\trace_0
]trace_1
^trace_2
_trace_32╩
@__inference_policy_layer_call_and_return_conditional_losses_1470
@__inference_policy_layer_call_and_return_conditional_losses_1591
@__inference_policy_layer_call_and_return_conditional_losses_1194
@__inference_policy_layer_call_and_return_conditional_losses_1236┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z\trace_0z]trace_1z^trace_2z_trace_3
╔B╞
__inference__wrapped_model_687input_2"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
,
`serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ы
ftrace_02╬
'__inference_conv2d_3_layer_call_fn_1600в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zftrace_0
Ж
gtrace_02щ
B__inference_conv2d_3_layer_call_and_return_conditional_losses_1633в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zgtrace_0
):' 2conv2d_3/kernel
: 2conv2d_3/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
ы
mtrace_02╬
'__inference_conv2d_4_layer_call_fn_1642в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zmtrace_0
Ж
ntrace_02щ
B__inference_conv2d_4_layer_call_and_return_conditional_losses_1675в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zntrace_0
):' @2conv2d_4/kernel
:@2conv2d_4/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
ы
ttrace_02╬
'__inference_conv2d_5_layer_call_fn_1684в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zttrace_0
Ж
utrace_02щ
B__inference_conv2d_5_layer_call_and_return_conditional_losses_1717в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zutrace_0
):'@@2conv2d_5/kernel
:@2conv2d_5/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
ь
{trace_02╧
(__inference_flatten_1_layer_call_fn_1722в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z{trace_0
З
|trace_02ъ
C__inference_flatten_1_layer_call_and_return_conditional_losses_1728в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z|trace_0
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
п
}non_trainable_variables

~layers
metrics
 Аlayer_regularization_losses
Бlayer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
ь
Вtrace_02═
&__inference_dense_1_layer_call_fn_1737в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zВtrace_0
З
Гtrace_02ш
A__inference_dense_1_layer_call_and_return_conditional_losses_1748в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zГtrace_0
": 
АbА2dense_1/kernel
:А2dense_1/bias
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Дnon_trainable_variables
Еlayers
Жmetrics
 Зlayer_regularization_losses
Иlayer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
ъ
Йtrace_02╦
$__inference_actor_layer_call_fn_1757в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЙtrace_0
Е
Кtrace_02ц
?__inference_actor_layer_call_and_return_conditional_losses_1768в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zКtrace_0
:	А2actor/kernel
:2
actor/bias
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Лnon_trainable_variables
Мlayers
Нmetrics
 Оlayer_regularization_losses
Пlayer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
я
Рtrace_02╨
)__inference_critic_ext_layer_call_fn_1777в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zРtrace_0
К
Сtrace_02ы
D__inference_critic_ext_layer_call_and_return_conditional_losses_1787в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zСtrace_0
$:"	А2critic_ext/kernel
:2critic_ext/bias
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Тnon_trainable_variables
Уlayers
Фmetrics
 Хlayer_regularization_losses
Цlayer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
я
Чtrace_02╨
)__inference_critic_int_layer_call_fn_1796в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЧtrace_0
К
Шtrace_02ы
D__inference_critic_int_layer_call_and_return_conditional_losses_1806в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zШtrace_0
$:"	А2critic_int/kernel
:2critic_int/bias
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЎBє
$__inference_policy_layer_call_fn_923input_2"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЎBє
%__inference_policy_layer_call_fn_1312inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЎBє
%__inference_policy_layer_call_fn_1349inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
%__inference_policy_layer_call_fn_1152input_2"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
СBО
@__inference_policy_layer_call_and_return_conditional_losses_1470inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
СBО
@__inference_policy_layer_call_and_return_conditional_losses_1591inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ТBП
@__inference_policy_layer_call_and_return_conditional_losses_1194input_2"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ТBП
@__inference_policy_layer_call_and_return_conditional_losses_1236input_2"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╔B╞
"__inference_signature_wrapper_1275input_2"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
█B╪
'__inference_conv2d_3_layer_call_fn_1600inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЎBє
B__inference_conv2d_3_layer_call_and_return_conditional_losses_1633inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
█B╪
'__inference_conv2d_4_layer_call_fn_1642inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЎBє
B__inference_conv2d_4_layer_call_and_return_conditional_losses_1675inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
█B╪
'__inference_conv2d_5_layer_call_fn_1684inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЎBє
B__inference_conv2d_5_layer_call_and_return_conditional_losses_1717inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▄B┘
(__inference_flatten_1_layer_call_fn_1722inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
C__inference_flatten_1_layer_call_and_return_conditional_losses_1728inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
┌B╫
&__inference_dense_1_layer_call_fn_1737inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
їBЄ
A__inference_dense_1_layer_call_and_return_conditional_losses_1748inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╪B╒
$__inference_actor_layer_call_fn_1757inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
єBЁ
?__inference_actor_layer_call_and_return_conditional_losses_1768inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▌B┌
)__inference_critic_ext_layer_call_fn_1777inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
°Bї
D__inference_critic_ext_layer_call_and_return_conditional_losses_1787inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▌B┌
)__inference_critic_int_layer_call_fn_1796inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
°Bї
D__inference_critic_int_layer_call_and_return_conditional_losses_1806inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 К
__inference__wrapped_model_687ч!"*+9:QRIJAB<в9
2в/
-К*
input_2         TT
к "ЦкТ
(
actorК
actor         
2

critic_ext$К!

critic_ext         
2

critic_int$К!

critic_int         з
?__inference_actor_layer_call_and_return_conditional_losses_1768dAB0в-
&в#
!К
inputs         А
к ",в)
"К
tensor_0         
Ъ Б
$__inference_actor_layer_call_fn_1757YAB0в-
&в#
!К
inputs         А
к "!К
unknown         ┴
B__inference_conv2d_3_layer_call_and_return_conditional_losses_1633{;в8
1в.
,К)
inputs         TT
к "8в5
.К+
tensor_0          
Ъ Ы
'__inference_conv2d_3_layer_call_fn_1600p;в8
1в.
,К)
inputs         TT
к "-К*
unknown          ┴
B__inference_conv2d_4_layer_call_and_return_conditional_losses_1675{!";в8
1в.
,К)
inputs          
к "8в5
.К+
tensor_0         		@
Ъ Ы
'__inference_conv2d_4_layer_call_fn_1642p!";в8
1в.
,К)
inputs          
к "-К*
unknown         		@┴
B__inference_conv2d_5_layer_call_and_return_conditional_losses_1717{*+;в8
1в.
,К)
inputs         		@
к "8в5
.К+
tensor_0         @
Ъ Ы
'__inference_conv2d_5_layer_call_fn_1684p*+;в8
1в.
,К)
inputs         		@
к "-К*
unknown         @м
D__inference_critic_ext_layer_call_and_return_conditional_losses_1787dIJ0в-
&в#
!К
inputs         А
к ",в)
"К
tensor_0         
Ъ Ж
)__inference_critic_ext_layer_call_fn_1777YIJ0в-
&в#
!К
inputs         А
к "!К
unknown         м
D__inference_critic_int_layer_call_and_return_conditional_losses_1806dQR0в-
&в#
!К
inputs         А
к ",в)
"К
tensor_0         
Ъ Ж
)__inference_critic_int_layer_call_fn_1796YQR0в-
&в#
!К
inputs         А
к "!К
unknown         к
A__inference_dense_1_layer_call_and_return_conditional_losses_1748e9:0в-
&в#
!К
inputs         Аb
к "-в*
#К 
tensor_0         А
Ъ Д
&__inference_dense_1_layer_call_fn_1737Z9:0в-
&в#
!К
inputs         Аb
к ""К
unknown         А│
C__inference_flatten_1_layer_call_and_return_conditional_losses_1728l;в8
1в.
,К)
inputs         @
к "-в*
#К 
tensor_0         Аb
Ъ Н
(__inference_flatten_1_layer_call_fn_1722a;в8
1в.
,К)
inputs         @
к ""К
unknown         АbЬ
@__inference_policy_layer_call_and_return_conditional_losses_1194╫!"*+9:QRIJABDвA
:в7
-К*
input_2         TT
p 

 
к "в|
uЪr
$К!

tensor_0_0         
$К!

tensor_0_1         
$К!

tensor_0_2         
Ъ Ь
@__inference_policy_layer_call_and_return_conditional_losses_1236╫!"*+9:QRIJABDвA
:в7
-К*
input_2         TT
p

 
к "в|
uЪr
$К!

tensor_0_0         
$К!

tensor_0_1         
$К!

tensor_0_2         
Ъ Ы
@__inference_policy_layer_call_and_return_conditional_losses_1470╓!"*+9:QRIJABCв@
9в6
,К)
inputs         TT
p 

 
к "в|
uЪr
$К!

tensor_0_0         
$К!

tensor_0_1         
$К!

tensor_0_2         
Ъ Ы
@__inference_policy_layer_call_and_return_conditional_losses_1591╓!"*+9:QRIJABCв@
9в6
,К)
inputs         TT
p

 
к "в|
uЪr
$К!

tensor_0_0         
$К!

tensor_0_1         
$К!

tensor_0_2         
Ъ ё
%__inference_policy_layer_call_fn_1152╟!"*+9:QRIJABDвA
:в7
-К*
input_2         TT
p

 
к "oЪl
"К
tensor_0         
"К
tensor_1         
"К
tensor_2         Ё
%__inference_policy_layer_call_fn_1312╞!"*+9:QRIJABCв@
9в6
,К)
inputs         TT
p 

 
к "oЪl
"К
tensor_0         
"К
tensor_1         
"К
tensor_2         Ё
%__inference_policy_layer_call_fn_1349╞!"*+9:QRIJABCв@
9в6
,К)
inputs         TT
p

 
к "oЪl
"К
tensor_0         
"К
tensor_1         
"К
tensor_2         Ё
$__inference_policy_layer_call_fn_923╟!"*+9:QRIJABDвA
:в7
-К*
input_2         TT
p 

 
к "oЪl
"К
tensor_0         
"К
tensor_1         
"К
tensor_2         Щ
"__inference_signature_wrapper_1275Є!"*+9:QRIJABGвD
в 
=к:
8
input_2-К*
input_2         TT"ЦкТ
(
actorК
actor         
2

critic_ext$К!

critic_ext         
2

critic_int$К!

critic_int         