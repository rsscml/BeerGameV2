
δ΅
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
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
dtypetype
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
<
Selu
features"T
activations"T"
Ttype:
2
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
Α
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68ηΠ
§
%mlp_td3_model/policy/mlp/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*6
shared_name'%mlp_td3_model/policy/mlp/dense/kernel
 
9mlp_td3_model/policy/mlp/dense/kernel/Read/ReadVariableOpReadVariableOp%mlp_td3_model/policy/mlp/dense/kernel*
_output_shapes
:	?*
dtype0

#mlp_td3_model/policy/mlp/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#mlp_td3_model/policy/mlp/dense/bias

7mlp_td3_model/policy/mlp/dense/bias/Read/ReadVariableOpReadVariableOp#mlp_td3_model/policy/mlp/dense/bias*
_output_shapes	
:*
dtype0
¬
'mlp_td3_model/policy/mlp/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*8
shared_name)'mlp_td3_model/policy/mlp/dense_1/kernel
₯
;mlp_td3_model/policy/mlp/dense_1/kernel/Read/ReadVariableOpReadVariableOp'mlp_td3_model/policy/mlp/dense_1/kernel* 
_output_shapes
:
*
dtype0
£
%mlp_td3_model/policy/mlp/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%mlp_td3_model/policy/mlp/dense_1/bias

9mlp_td3_model/policy/mlp/dense_1/bias/Read/ReadVariableOpReadVariableOp%mlp_td3_model/policy/mlp/dense_1/bias*
_output_shapes	
:*
dtype0
¬
'mlp_td3_model/policy/mlp/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*8
shared_name)'mlp_td3_model/policy/mlp/dense_2/kernel
₯
;mlp_td3_model/policy/mlp/dense_2/kernel/Read/ReadVariableOpReadVariableOp'mlp_td3_model/policy/mlp/dense_2/kernel* 
_output_shapes
:
*
dtype0
£
%mlp_td3_model/policy/mlp/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%mlp_td3_model/policy/mlp/dense_2/bias

9mlp_td3_model/policy/mlp/dense_2/bias/Read/ReadVariableOpReadVariableOp%mlp_td3_model/policy/mlp/dense_2/bias*
_output_shapes	
:*
dtype0
«
'mlp_td3_model/policy/mlp/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Z*8
shared_name)'mlp_td3_model/policy/mlp/dense_3/kernel
€
;mlp_td3_model/policy/mlp/dense_3/kernel/Read/ReadVariableOpReadVariableOp'mlp_td3_model/policy/mlp/dense_3/kernel*
_output_shapes
:	Z*
dtype0
’
%mlp_td3_model/policy/mlp/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*6
shared_name'%mlp_td3_model/policy/mlp/dense_3/bias

9mlp_td3_model/policy/mlp/dense_3/bias/Read/ReadVariableOpReadVariableOp%mlp_td3_model/policy/mlp/dense_3/bias*
_output_shapes
:Z*
dtype0
¦
$mlp_td3_model/q/mlp_1/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*5
shared_name&$mlp_td3_model/q/mlp_1/dense_4/kernel

8mlp_td3_model/q/mlp_1/dense_4/kernel/Read/ReadVariableOpReadVariableOp$mlp_td3_model/q/mlp_1/dense_4/kernel* 
_output_shapes
:
*
dtype0

"mlp_td3_model/q/mlp_1/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"mlp_td3_model/q/mlp_1/dense_4/bias

6mlp_td3_model/q/mlp_1/dense_4/bias/Read/ReadVariableOpReadVariableOp"mlp_td3_model/q/mlp_1/dense_4/bias*
_output_shapes	
:*
dtype0
¦
$mlp_td3_model/q/mlp_1/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*5
shared_name&$mlp_td3_model/q/mlp_1/dense_5/kernel

8mlp_td3_model/q/mlp_1/dense_5/kernel/Read/ReadVariableOpReadVariableOp$mlp_td3_model/q/mlp_1/dense_5/kernel* 
_output_shapes
:
*
dtype0

"mlp_td3_model/q/mlp_1/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"mlp_td3_model/q/mlp_1/dense_5/bias

6mlp_td3_model/q/mlp_1/dense_5/bias/Read/ReadVariableOpReadVariableOp"mlp_td3_model/q/mlp_1/dense_5/bias*
_output_shapes	
:*
dtype0
¦
$mlp_td3_model/q/mlp_1/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*5
shared_name&$mlp_td3_model/q/mlp_1/dense_6/kernel

8mlp_td3_model/q/mlp_1/dense_6/kernel/Read/ReadVariableOpReadVariableOp$mlp_td3_model/q/mlp_1/dense_6/kernel* 
_output_shapes
:
*
dtype0

"mlp_td3_model/q/mlp_1/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"mlp_td3_model/q/mlp_1/dense_6/bias

6mlp_td3_model/q/mlp_1/dense_6/bias/Read/ReadVariableOpReadVariableOp"mlp_td3_model/q/mlp_1/dense_6/bias*
_output_shapes	
:*
dtype0
₯
$mlp_td3_model/q/mlp_1/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*5
shared_name&$mlp_td3_model/q/mlp_1/dense_7/kernel

8mlp_td3_model/q/mlp_1/dense_7/kernel/Read/ReadVariableOpReadVariableOp$mlp_td3_model/q/mlp_1/dense_7/kernel*
_output_shapes
:	*
dtype0

"mlp_td3_model/q/mlp_1/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"mlp_td3_model/q/mlp_1/dense_7/bias

6mlp_td3_model/q/mlp_1/dense_7/bias/Read/ReadVariableOpReadVariableOp"mlp_td3_model/q/mlp_1/dense_7/bias*
_output_shapes
:*
dtype0
ͺ
&mlp_td3_model/q_1/mlp_2/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*7
shared_name(&mlp_td3_model/q_1/mlp_2/dense_8/kernel
£
:mlp_td3_model/q_1/mlp_2/dense_8/kernel/Read/ReadVariableOpReadVariableOp&mlp_td3_model/q_1/mlp_2/dense_8/kernel* 
_output_shapes
:
*
dtype0
‘
$mlp_td3_model/q_1/mlp_2/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$mlp_td3_model/q_1/mlp_2/dense_8/bias

8mlp_td3_model/q_1/mlp_2/dense_8/bias/Read/ReadVariableOpReadVariableOp$mlp_td3_model/q_1/mlp_2/dense_8/bias*
_output_shapes	
:*
dtype0
ͺ
&mlp_td3_model/q_1/mlp_2/dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*7
shared_name(&mlp_td3_model/q_1/mlp_2/dense_9/kernel
£
:mlp_td3_model/q_1/mlp_2/dense_9/kernel/Read/ReadVariableOpReadVariableOp&mlp_td3_model/q_1/mlp_2/dense_9/kernel* 
_output_shapes
:
*
dtype0
‘
$mlp_td3_model/q_1/mlp_2/dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$mlp_td3_model/q_1/mlp_2/dense_9/bias

8mlp_td3_model/q_1/mlp_2/dense_9/bias/Read/ReadVariableOpReadVariableOp$mlp_td3_model/q_1/mlp_2/dense_9/bias*
_output_shapes	
:*
dtype0
¬
'mlp_td3_model/q_1/mlp_2/dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*8
shared_name)'mlp_td3_model/q_1/mlp_2/dense_10/kernel
₯
;mlp_td3_model/q_1/mlp_2/dense_10/kernel/Read/ReadVariableOpReadVariableOp'mlp_td3_model/q_1/mlp_2/dense_10/kernel* 
_output_shapes
:
*
dtype0
£
%mlp_td3_model/q_1/mlp_2/dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%mlp_td3_model/q_1/mlp_2/dense_10/bias

9mlp_td3_model/q_1/mlp_2/dense_10/bias/Read/ReadVariableOpReadVariableOp%mlp_td3_model/q_1/mlp_2/dense_10/bias*
_output_shapes	
:*
dtype0
«
'mlp_td3_model/q_1/mlp_2/dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*8
shared_name)'mlp_td3_model/q_1/mlp_2/dense_11/kernel
€
;mlp_td3_model/q_1/mlp_2/dense_11/kernel/Read/ReadVariableOpReadVariableOp'mlp_td3_model/q_1/mlp_2/dense_11/kernel*
_output_shapes
:	*
dtype0
’
%mlp_td3_model/q_1/mlp_2/dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%mlp_td3_model/q_1/mlp_2/dense_11/bias

9mlp_td3_model/q_1/mlp_2/dense_11/bias/Read/ReadVariableOpReadVariableOp%mlp_td3_model/q_1/mlp_2/dense_11/bias*
_output_shapes
:*
dtype0

NoOpNoOp
Φk
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*k
valuekBk Bύj
Υ
pi
q1
q2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature

signatures*

pi
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*

q
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*

q
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses*
Ί
!0
"1
#2
$3
%4
&5
'6
(7
)8
*9
+10
,11
-12
.13
/14
015
116
217
318
419
520
621
722
823*
Ί
!0
"1
#2
$3
%4
&5
'6
(7
)8
*9
+10
,11
-12
.13
/14
015
116
217
318
419
520
621
722
823*
* 
°
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*
* 
* 
* 

>serving_default* 
±
?dense_layers
@	dense_out
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses*
<
!0
"1
#2
$3
%4
&5
'6
(7*
<
!0
"1
#2
$3
%4
&5
'6
(7*
* 

Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
±
Ldense_layers
M	dense_out
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses*
<
)0
*1
+2
,3
-4
.5
/6
07*
<
)0
*1
+2
,3
-4
.5
/6
07*
* 

Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
±
Ydense_layers
Z	dense_out
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses*
<
10
21
32
43
54
65
76
87*
<
10
21
32
43
54
65
76
87*
* 

anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses*
* 
* 
e_
VARIABLE_VALUE%mlp_td3_model/policy/mlp/dense/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#mlp_td3_model/policy/mlp/dense/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'mlp_td3_model/policy/mlp/dense_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%mlp_td3_model/policy/mlp/dense_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'mlp_td3_model/policy/mlp/dense_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%mlp_td3_model/policy/mlp/dense_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'mlp_td3_model/policy/mlp/dense_3/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%mlp_td3_model/policy/mlp/dense_3/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE$mlp_td3_model/q/mlp_1/dense_4/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"mlp_td3_model/q/mlp_1/dense_4/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE$mlp_td3_model/q/mlp_1/dense_5/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"mlp_td3_model/q/mlp_1/dense_5/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE$mlp_td3_model/q/mlp_1/dense_6/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"mlp_td3_model/q/mlp_1/dense_6/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE$mlp_td3_model/q/mlp_1/dense_7/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"mlp_td3_model/q/mlp_1/dense_7/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&mlp_td3_model/q_1/mlp_2/dense_8/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE$mlp_td3_model/q_1/mlp_2/dense_8/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&mlp_td3_model/q_1/mlp_2/dense_9/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE$mlp_td3_model/q_1/mlp_2/dense_9/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUE'mlp_td3_model/q_1/mlp_2/dense_10/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%mlp_td3_model/q_1/mlp_2/dense_10/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUE'mlp_td3_model/q_1/mlp_2/dense_11/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%mlp_td3_model/q_1/mlp_2/dense_11/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*
* 
* 
* 
* 

f0
g1
h2*
¦

'kernel
(bias
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses*
<
!0
"1
#2
$3
%4
&5
'6
(7*
<
!0
"1
#2
$3
%4
&5
'6
(7*
* 

onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses*
* 
* 
* 

0*
* 
* 
* 

t0
u1
v2*
¦

/kernel
0bias
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses*
<
)0
*1
+2
,3
-4
.5
/6
07*
<
)0
*1
+2
,3
-4
.5
/6
07*
* 

}non_trainable_variables

~layers
metrics
 layer_regularization_losses
layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses*
* 
* 
* 

0*
* 
* 
* 

0
1
2*
¬

7kernel
8bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
<
10
21
32
43
54
65
76
87*
<
10
21
32
43
54
65
76
87*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses*
* 
* 
* 

0*
* 
* 
* 
¬

!kernel
"bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
¬

#kernel
$bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
¬

%kernel
&bias
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
+‘&call_and_return_all_conditional_losses*

'0
(1*

'0
(1*
* 

’non_trainable_variables
£layers
€metrics
 ₯layer_regularization_losses
¦layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses*
* 
* 
* 
 
f0
g1
h2
@3*
* 
* 
* 
¬

)kernel
*bias
§	variables
¨trainable_variables
©regularization_losses
ͺ	keras_api
«__call__
+¬&call_and_return_all_conditional_losses*
¬

+kernel
,bias
­	variables
?trainable_variables
―regularization_losses
°	keras_api
±__call__
+²&call_and_return_all_conditional_losses*
¬

-kernel
.bias
³	variables
΄trainable_variables
΅regularization_losses
Ά	keras_api
·__call__
+Έ&call_and_return_all_conditional_losses*

/0
01*

/0
01*
* 

Ήnon_trainable_variables
Ίlayers
»metrics
 Όlayer_regularization_losses
½layer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses*
* 
* 
* 
 
t0
u1
v2
M3*
* 
* 
* 
¬

1kernel
2bias
Ύ	variables
Ώtrainable_variables
ΐregularization_losses
Α	keras_api
Β__call__
+Γ&call_and_return_all_conditional_losses*
¬

3kernel
4bias
Δ	variables
Εtrainable_variables
Ζregularization_losses
Η	keras_api
Θ__call__
+Ι&call_and_return_all_conditional_losses*
¬

5kernel
6bias
Κ	variables
Λtrainable_variables
Μregularization_losses
Ν	keras_api
Ξ__call__
+Ο&call_and_return_all_conditional_losses*

70
81*

70
81*
* 

Πnon_trainable_variables
Ρlayers
?metrics
 Σlayer_regularization_losses
Τlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
#
0
1
2
Z3*
* 
* 
* 

!0
"1*

!0
"1*
* 

Υnon_trainable_variables
Φlayers
Χmetrics
 Ψlayer_regularization_losses
Ωlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 

#0
$1*

#0
$1*
* 

Ϊnon_trainable_variables
Ϋlayers
άmetrics
 έlayer_regularization_losses
ήlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 

%0
&1*

%0
&1*
* 

ίnon_trainable_variables
ΰlayers
αmetrics
 βlayer_regularization_losses
γlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
+‘&call_and_return_all_conditional_losses
'‘"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 

)0
*1*

)0
*1*
* 

δnon_trainable_variables
εlayers
ζmetrics
 ηlayer_regularization_losses
θlayer_metrics
§	variables
¨trainable_variables
©regularization_losses
«__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses*
* 
* 

+0
,1*

+0
,1*
* 

ιnon_trainable_variables
κlayers
λmetrics
 μlayer_regularization_losses
νlayer_metrics
­	variables
?trainable_variables
―regularization_losses
±__call__
+²&call_and_return_all_conditional_losses
'²"call_and_return_conditional_losses*
* 
* 

-0
.1*

-0
.1*
* 

ξnon_trainable_variables
οlayers
πmetrics
 ρlayer_regularization_losses
ςlayer_metrics
³	variables
΄trainable_variables
΅regularization_losses
·__call__
+Έ&call_and_return_all_conditional_losses
'Έ"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 

10
21*

10
21*
* 

σnon_trainable_variables
τlayers
υmetrics
 φlayer_regularization_losses
χlayer_metrics
Ύ	variables
Ώtrainable_variables
ΐregularization_losses
Β__call__
+Γ&call_and_return_all_conditional_losses
'Γ"call_and_return_conditional_losses*
* 
* 

30
41*

30
41*
* 

ψnon_trainable_variables
ωlayers
ϊmetrics
 ϋlayer_regularization_losses
όlayer_metrics
Δ	variables
Εtrainable_variables
Ζregularization_losses
Θ__call__
+Ι&call_and_return_all_conditional_losses
'Ι"call_and_return_conditional_losses*
* 
* 

50
61*

50
61*
* 

ύnon_trainable_variables
ώlayers
?metrics
 layer_regularization_losses
layer_metrics
Κ	variables
Λtrainable_variables
Μregularization_losses
Ξ__call__
+Ο&call_and_return_all_conditional_losses
'Ο"call_and_return_conditional_losses*
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
z
serving_default_input_1Placeholder*'
_output_shapes
:??????????*
dtype0*
shape:??????????
z
serving_default_input_2Placeholder*'
_output_shapes
:?????????Z*
dtype0*
shape:?????????Z
Ϋ

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2%mlp_td3_model/policy/mlp/dense/kernel#mlp_td3_model/policy/mlp/dense/bias'mlp_td3_model/policy/mlp/dense_1/kernel%mlp_td3_model/policy/mlp/dense_1/bias'mlp_td3_model/policy/mlp/dense_2/kernel%mlp_td3_model/policy/mlp/dense_2/bias'mlp_td3_model/policy/mlp/dense_3/kernel%mlp_td3_model/policy/mlp/dense_3/bias$mlp_td3_model/q/mlp_1/dense_4/kernel"mlp_td3_model/q/mlp_1/dense_4/bias$mlp_td3_model/q/mlp_1/dense_5/kernel"mlp_td3_model/q/mlp_1/dense_5/bias$mlp_td3_model/q/mlp_1/dense_6/kernel"mlp_td3_model/q/mlp_1/dense_6/bias$mlp_td3_model/q/mlp_1/dense_7/kernel"mlp_td3_model/q/mlp_1/dense_7/bias&mlp_td3_model/q_1/mlp_2/dense_8/kernel$mlp_td3_model/q_1/mlp_2/dense_8/bias&mlp_td3_model/q_1/mlp_2/dense_9/kernel$mlp_td3_model/q_1/mlp_2/dense_9/bias'mlp_td3_model/q_1/mlp_2/dense_10/kernel%mlp_td3_model/q_1/mlp_2/dense_10/bias'mlp_td3_model/q_1/mlp_2/dense_11/kernel%mlp_td3_model/q_1/mlp_2/dense_11/bias*%
Tin
2*
Tout
2*
_collective_manager_ids
 *`
_output_shapesN
L:?????????Z:?????????:?????????:?????????*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_signature_wrapper_178346112
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ά
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename9mlp_td3_model/policy/mlp/dense/kernel/Read/ReadVariableOp7mlp_td3_model/policy/mlp/dense/bias/Read/ReadVariableOp;mlp_td3_model/policy/mlp/dense_1/kernel/Read/ReadVariableOp9mlp_td3_model/policy/mlp/dense_1/bias/Read/ReadVariableOp;mlp_td3_model/policy/mlp/dense_2/kernel/Read/ReadVariableOp9mlp_td3_model/policy/mlp/dense_2/bias/Read/ReadVariableOp;mlp_td3_model/policy/mlp/dense_3/kernel/Read/ReadVariableOp9mlp_td3_model/policy/mlp/dense_3/bias/Read/ReadVariableOp8mlp_td3_model/q/mlp_1/dense_4/kernel/Read/ReadVariableOp6mlp_td3_model/q/mlp_1/dense_4/bias/Read/ReadVariableOp8mlp_td3_model/q/mlp_1/dense_5/kernel/Read/ReadVariableOp6mlp_td3_model/q/mlp_1/dense_5/bias/Read/ReadVariableOp8mlp_td3_model/q/mlp_1/dense_6/kernel/Read/ReadVariableOp6mlp_td3_model/q/mlp_1/dense_6/bias/Read/ReadVariableOp8mlp_td3_model/q/mlp_1/dense_7/kernel/Read/ReadVariableOp6mlp_td3_model/q/mlp_1/dense_7/bias/Read/ReadVariableOp:mlp_td3_model/q_1/mlp_2/dense_8/kernel/Read/ReadVariableOp8mlp_td3_model/q_1/mlp_2/dense_8/bias/Read/ReadVariableOp:mlp_td3_model/q_1/mlp_2/dense_9/kernel/Read/ReadVariableOp8mlp_td3_model/q_1/mlp_2/dense_9/bias/Read/ReadVariableOp;mlp_td3_model/q_1/mlp_2/dense_10/kernel/Read/ReadVariableOp9mlp_td3_model/q_1/mlp_2/dense_10/bias/Read/ReadVariableOp;mlp_td3_model/q_1/mlp_2/dense_11/kernel/Read/ReadVariableOp9mlp_td3_model/q_1/mlp_2/dense_11/bias/Read/ReadVariableOpConst*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_save_178346855
Ρ	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename%mlp_td3_model/policy/mlp/dense/kernel#mlp_td3_model/policy/mlp/dense/bias'mlp_td3_model/policy/mlp/dense_1/kernel%mlp_td3_model/policy/mlp/dense_1/bias'mlp_td3_model/policy/mlp/dense_2/kernel%mlp_td3_model/policy/mlp/dense_2/bias'mlp_td3_model/policy/mlp/dense_3/kernel%mlp_td3_model/policy/mlp/dense_3/bias$mlp_td3_model/q/mlp_1/dense_4/kernel"mlp_td3_model/q/mlp_1/dense_4/bias$mlp_td3_model/q/mlp_1/dense_5/kernel"mlp_td3_model/q/mlp_1/dense_5/bias$mlp_td3_model/q/mlp_1/dense_6/kernel"mlp_td3_model/q/mlp_1/dense_6/bias$mlp_td3_model/q/mlp_1/dense_7/kernel"mlp_td3_model/q/mlp_1/dense_7/bias&mlp_td3_model/q_1/mlp_2/dense_8/kernel$mlp_td3_model/q_1/mlp_2/dense_8/bias&mlp_td3_model/q_1/mlp_2/dense_9/kernel$mlp_td3_model/q_1/mlp_2/dense_9/bias'mlp_td3_model/q_1/mlp_2/dense_10/kernel%mlp_td3_model/q_1/mlp_2/dense_10/bias'mlp_td3_model/q_1/mlp_2/dense_11/kernel%mlp_td3_model/q_1/mlp_2/dense_11/bias*$
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference__traced_restore_178346937?
Μ	
Γ
*__inference_policy_layer_call_fn_178344401
input_1
unknown:	?
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	Z
	unknown_6:Z
identity’StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_policy_layer_call_and_return_conditional_losses_178344382o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:??????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:??????????
!
_user_specified_name	input_1
β)

@__inference_q_layer_call_and_return_conditional_losses_178346330	
inp_0	
inp_1@
,mlp_1_dense_4_matmul_readvariableop_resource:
<
-mlp_1_dense_4_biasadd_readvariableop_resource:	@
,mlp_1_dense_5_matmul_readvariableop_resource:
<
-mlp_1_dense_5_biasadd_readvariableop_resource:	@
,mlp_1_dense_6_matmul_readvariableop_resource:
<
-mlp_1_dense_6_biasadd_readvariableop_resource:	?
,mlp_1_dense_7_matmul_readvariableop_resource:	;
-mlp_1_dense_7_biasadd_readvariableop_resource:
identity’$mlp_1/dense_4/BiasAdd/ReadVariableOp’#mlp_1/dense_4/MatMul/ReadVariableOp’$mlp_1/dense_5/BiasAdd/ReadVariableOp’#mlp_1/dense_5/MatMul/ReadVariableOp’$mlp_1/dense_6/BiasAdd/ReadVariableOp’#mlp_1/dense_6/MatMul/ReadVariableOp’$mlp_1/dense_7/BiasAdd/ReadVariableOp’#mlp_1/dense_7/MatMul/ReadVariableOpV
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????r
concatConcatV2inp_0inp_1concat/axis:output:0*
N*
T0*(
_output_shapes
:?????????
#mlp_1/dense_4/MatMul/ReadVariableOpReadVariableOp,mlp_1_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
mlp_1/dense_4/MatMulMatMulconcat:output:0+mlp_1/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
$mlp_1/dense_4/BiasAdd/ReadVariableOpReadVariableOp-mlp_1_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0‘
mlp_1/dense_4/BiasAddBiasAddmlp_1/dense_4/MatMul:product:0,mlp_1/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????m
mlp_1/dense_4/SeluSelumlp_1/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
#mlp_1/dense_5/MatMul/ReadVariableOpReadVariableOp,mlp_1_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0 
mlp_1/dense_5/MatMulMatMul mlp_1/dense_4/Selu:activations:0+mlp_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
$mlp_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp-mlp_1_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0‘
mlp_1/dense_5/BiasAddBiasAddmlp_1/dense_5/MatMul:product:0,mlp_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????m
mlp_1/dense_5/SeluSelumlp_1/dense_5/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
#mlp_1/dense_6/MatMul/ReadVariableOpReadVariableOp,mlp_1_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0 
mlp_1/dense_6/MatMulMatMul mlp_1/dense_5/Selu:activations:0+mlp_1/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
$mlp_1/dense_6/BiasAdd/ReadVariableOpReadVariableOp-mlp_1_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0‘
mlp_1/dense_6/BiasAddBiasAddmlp_1/dense_6/MatMul:product:0,mlp_1/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????m
mlp_1/dense_6/SeluSelumlp_1/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
#mlp_1/dense_7/MatMul/ReadVariableOpReadVariableOp,mlp_1_dense_7_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
mlp_1/dense_7/MatMulMatMul mlp_1/dense_6/Selu:activations:0+mlp_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
$mlp_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp-mlp_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0 
mlp_1/dense_7/BiasAddBiasAddmlp_1/dense_7/MatMul:product:0,mlp_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m
IdentityIdentitymlp_1/dense_7/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????ϊ
NoOpNoOp%^mlp_1/dense_4/BiasAdd/ReadVariableOp$^mlp_1/dense_4/MatMul/ReadVariableOp%^mlp_1/dense_5/BiasAdd/ReadVariableOp$^mlp_1/dense_5/MatMul/ReadVariableOp%^mlp_1/dense_6/BiasAdd/ReadVariableOp$^mlp_1/dense_6/MatMul/ReadVariableOp%^mlp_1/dense_7/BiasAdd/ReadVariableOp$^mlp_1/dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:??????????:?????????Z: : : : : : : : 2L
$mlp_1/dense_4/BiasAdd/ReadVariableOp$mlp_1/dense_4/BiasAdd/ReadVariableOp2J
#mlp_1/dense_4/MatMul/ReadVariableOp#mlp_1/dense_4/MatMul/ReadVariableOp2L
$mlp_1/dense_5/BiasAdd/ReadVariableOp$mlp_1/dense_5/BiasAdd/ReadVariableOp2J
#mlp_1/dense_5/MatMul/ReadVariableOp#mlp_1/dense_5/MatMul/ReadVariableOp2L
$mlp_1/dense_6/BiasAdd/ReadVariableOp$mlp_1/dense_6/BiasAdd/ReadVariableOp2J
#mlp_1/dense_6/MatMul/ReadVariableOp#mlp_1/dense_6/MatMul/ReadVariableOp2L
$mlp_1/dense_7/BiasAdd/ReadVariableOp$mlp_1/dense_7/BiasAdd/ReadVariableOp2J
#mlp_1/dense_7/MatMul/ReadVariableOp#mlp_1/dense_7/MatMul/ReadVariableOp:N J
'
_output_shapes
:??????????

_user_specified_nameinp/0:NJ
'
_output_shapes
:?????????Z

_user_specified_nameinp/1
Γ#
Ί
D__inference_mlp_2_layer_call_and_return_conditional_losses_178346756
obs:
&dense_8_matmul_readvariableop_resource:
6
'dense_8_biasadd_readvariableop_resource:	:
&dense_9_matmul_readvariableop_resource:
6
'dense_9_biasadd_readvariableop_resource:	;
'dense_10_matmul_readvariableop_resource:
7
(dense_10_biasadd_readvariableop_resource:	:
'dense_11_matmul_readvariableop_resource:	6
(dense_11_biasadd_readvariableop_resource:
identity’dense_10/BiasAdd/ReadVariableOp’dense_10/MatMul/ReadVariableOp’dense_11/BiasAdd/ReadVariableOp’dense_11/MatMul/ReadVariableOp’dense_8/BiasAdd/ReadVariableOp’dense_8/MatMul/ReadVariableOp’dense_9/BiasAdd/ReadVariableOp’dense_9/MatMul/ReadVariableOp
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0w
dense_8/MatMulMatMulobs%dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????a
dense_8/SeluSeludense_8/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_9/MatMulMatMuldense_8/Selu:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????a
dense_9/SeluSeludense_9/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_10/MatMulMatMuldense_9/Selu:activations:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????c
dense_10/SeluSeludense_10/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_11/MatMulMatMuldense_10/Selu:activations:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_11/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????Ξ
NoOpNoOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????: : : : : : : : 2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:M I
(
_output_shapes
:?????????

_user_specified_nameobs
Ζ
Τ
1__inference_mlp_td3_model_layer_call_fn_178345820	
inp_0	
inp_1
unknown:	?
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	Z
	unknown_6:Z
	unknown_7:

	unknown_8:	
	unknown_9:


unknown_10:	

unknown_11:


unknown_12:	

unknown_13:	

unknown_14:

unknown_15:


unknown_16:	

unknown_17:


unknown_18:	

unknown_19:


unknown_20:	

unknown_21:	

unknown_22:
identity

identity_1

identity_2

identity_3’StatefulPartitionedCallΟ
StatefulPartitionedCallStatefulPartitionedCallinp_0inp_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *`
_output_shapesN
L:?????????Z:?????????:?????????:?????????*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_mlp_td3_model_layer_call_and_return_conditional_losses_178345447o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Zq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:?????????q

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:??????????:?????????Z: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:??????????

_user_specified_nameinp/0:NJ
'
_output_shapes
:?????????Z

_user_specified_nameinp/1
³

Θ
%__inference_q_layer_call_fn_178346262	
inp_0	
inp_1
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	
	unknown_6:
identity’StatefulPartitionedCallͺ
StatefulPartitionedCallStatefulPartitionedCallinp_0inp_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_q_layer_call_and_return_conditional_losses_178344805o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:??????????:?????????Z: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:??????????

_user_specified_nameinp/0:NJ
'
_output_shapes
:?????????Z

_user_specified_nameinp/1
Ι#
§
B__inference_mlp_layer_call_and_return_conditional_losses_178346516
obs7
$dense_matmul_readvariableop_resource:	?4
%dense_biasadd_readvariableop_resource:	:
&dense_1_matmul_readvariableop_resource:
6
'dense_1_biasadd_readvariableop_resource:	:
&dense_2_matmul_readvariableop_resource:
6
'dense_2_biasadd_readvariableop_resource:	9
&dense_3_matmul_readvariableop_resource:	Z5
'dense_3_biasadd_readvariableop_resource:Z
identity’dense/BiasAdd/ReadVariableOp’dense/MatMul/ReadVariableOp’dense_1/BiasAdd/ReadVariableOp’dense_1/MatMul/ReadVariableOp’dense_2/BiasAdd/ReadVariableOp’dense_2/MatMul/ReadVariableOp’dense_3/BiasAdd/ReadVariableOp’dense_3/MatMul/ReadVariableOp
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0s
dense/MatMulMatMulobs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????]

dense/SeluSeludense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_1/MatMulMatMuldense/Selu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????a
dense_1/SeluSeludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_2/MatMulMatMuldense_1/Selu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????a
dense_2/SeluSeludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	Z*
dtype0
dense_3/MatMulMatMuldense_2/Selu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:Z*
dtype0
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Zf
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????Zb
IdentityIdentitydense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????ZΖ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:??????????: : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:L H
'
_output_shapes
:??????????

_user_specified_nameobs
?
Ψ
1__inference_mlp_td3_model_layer_call_fn_178345316
input_1
input_2
unknown:	?
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	Z
	unknown_6:Z
	unknown_7:

	unknown_8:	
	unknown_9:


unknown_10:	

unknown_11:


unknown_12:	

unknown_13:	

unknown_14:

unknown_15:


unknown_16:	

unknown_17:


unknown_18:	

unknown_19:


unknown_20:	

unknown_21:	

unknown_22:
identity

identity_1

identity_2

identity_3’StatefulPartitionedCallΣ
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *`
_output_shapesN
L:?????????Z:?????????:?????????:?????????*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_mlp_td3_model_layer_call_and_return_conditional_losses_178345259o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Zq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:?????????q

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:??????????:?????????Z: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:??????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????Z
!
_user_specified_name	input_2
­&
κ
E__inference_policy_layer_call_and_return_conditional_losses_178346186
obs;
(mlp_dense_matmul_readvariableop_resource:	?8
)mlp_dense_biasadd_readvariableop_resource:	>
*mlp_dense_1_matmul_readvariableop_resource:
:
+mlp_dense_1_biasadd_readvariableop_resource:	>
*mlp_dense_2_matmul_readvariableop_resource:
:
+mlp_dense_2_biasadd_readvariableop_resource:	=
*mlp_dense_3_matmul_readvariableop_resource:	Z9
+mlp_dense_3_biasadd_readvariableop_resource:Z
identity’ mlp/dense/BiasAdd/ReadVariableOp’mlp/dense/MatMul/ReadVariableOp’"mlp/dense_1/BiasAdd/ReadVariableOp’!mlp/dense_1/MatMul/ReadVariableOp’"mlp/dense_2/BiasAdd/ReadVariableOp’!mlp/dense_2/MatMul/ReadVariableOp’"mlp/dense_3/BiasAdd/ReadVariableOp’!mlp/dense_3/MatMul/ReadVariableOp
mlp/dense/MatMul/ReadVariableOpReadVariableOp(mlp_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0{
mlp/dense/MatMulMatMulobs'mlp/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
 mlp/dense/BiasAdd/ReadVariableOpReadVariableOp)mlp_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
mlp/dense/BiasAddBiasAddmlp/dense/MatMul:product:0(mlp/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????e
mlp/dense/SeluSelumlp/dense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
!mlp/dense_1/MatMul/ReadVariableOpReadVariableOp*mlp_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
mlp/dense_1/MatMulMatMulmlp/dense/Selu:activations:0)mlp/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
"mlp/dense_1/BiasAdd/ReadVariableOpReadVariableOp+mlp_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
mlp/dense_1/BiasAddBiasAddmlp/dense_1/MatMul:product:0*mlp/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????i
mlp/dense_1/SeluSelumlp/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
!mlp/dense_2/MatMul/ReadVariableOpReadVariableOp*mlp_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
mlp/dense_2/MatMulMatMulmlp/dense_1/Selu:activations:0)mlp/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
"mlp/dense_2/BiasAdd/ReadVariableOpReadVariableOp+mlp_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
mlp/dense_2/BiasAddBiasAddmlp/dense_2/MatMul:product:0*mlp/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????i
mlp/dense_2/SeluSelumlp/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
!mlp/dense_3/MatMul/ReadVariableOpReadVariableOp*mlp_dense_3_matmul_readvariableop_resource*
_output_shapes
:	Z*
dtype0
mlp/dense_3/MatMulMatMulmlp/dense_2/Selu:activations:0)mlp/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z
"mlp/dense_3/BiasAdd/ReadVariableOpReadVariableOp+mlp_dense_3_biasadd_readvariableop_resource*
_output_shapes
:Z*
dtype0
mlp/dense_3/BiasAddBiasAddmlp/dense_3/MatMul:product:0*mlp/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Zn
mlp/dense_3/SigmoidSigmoidmlp/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????Zf
IdentityIdentitymlp/dense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????Zζ
NoOpNoOp!^mlp/dense/BiasAdd/ReadVariableOp ^mlp/dense/MatMul/ReadVariableOp#^mlp/dense_1/BiasAdd/ReadVariableOp"^mlp/dense_1/MatMul/ReadVariableOp#^mlp/dense_2/BiasAdd/ReadVariableOp"^mlp/dense_2/MatMul/ReadVariableOp#^mlp/dense_3/BiasAdd/ReadVariableOp"^mlp/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:??????????: : : : : : : : 2D
 mlp/dense/BiasAdd/ReadVariableOp mlp/dense/BiasAdd/ReadVariableOp2B
mlp/dense/MatMul/ReadVariableOpmlp/dense/MatMul/ReadVariableOp2H
"mlp/dense_1/BiasAdd/ReadVariableOp"mlp/dense_1/BiasAdd/ReadVariableOp2F
!mlp/dense_1/MatMul/ReadVariableOp!mlp/dense_1/MatMul/ReadVariableOp2H
"mlp/dense_2/BiasAdd/ReadVariableOp"mlp/dense_2/BiasAdd/ReadVariableOp2F
!mlp/dense_2/MatMul/ReadVariableOp!mlp/dense_2/MatMul/ReadVariableOp2H
"mlp/dense_3/BiasAdd/ReadVariableOp"mlp/dense_3/BiasAdd/ReadVariableOp2F
!mlp/dense_3/MatMul/ReadVariableOp!mlp/dense_3/MatMul/ReadVariableOp:L H
'
_output_shapes
:??????????

_user_specified_nameobs
Ί	
Ό
'__inference_mlp_layer_call_fn_178346463
obs
unknown:	?
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	Z
	unknown_6:Z
identity’StatefulPartitionedCall’
StatefulPartitionedCallStatefulPartitionedCallobsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_mlp_layer_call_and_return_conditional_losses_178344363o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:??????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
'
_output_shapes
:??????????

_user_specified_nameobs
ΐ	
Ώ
*__inference_policy_layer_call_fn_178346133
obs
unknown:	?
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	Z
	unknown_6:Z
identity’StatefulPartitionedCall₯
StatefulPartitionedCallStatefulPartitionedCallobsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_policy_layer_call_and_return_conditional_losses_178344382o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:??????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
'
_output_shapes
:??????????

_user_specified_nameobs
·

Κ
'__inference_q_1_layer_call_fn_178346374	
inp_0	
inp_1
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	
	unknown_6:
identity’StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinp_0inp_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_q_1_layer_call_and_return_conditional_losses_178345097o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:??????????:?????????Z: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:??????????

_user_specified_nameinp/0:NJ
'
_output_shapes
:?????????Z

_user_specified_nameinp/1
Υ’
Ύ
L__inference_mlp_td3_model_layer_call_and_return_conditional_losses_178346050	
inp_0	
inp_1B
/policy_mlp_dense_matmul_readvariableop_resource:	??
0policy_mlp_dense_biasadd_readvariableop_resource:	E
1policy_mlp_dense_1_matmul_readvariableop_resource:
A
2policy_mlp_dense_1_biasadd_readvariableop_resource:	E
1policy_mlp_dense_2_matmul_readvariableop_resource:
A
2policy_mlp_dense_2_biasadd_readvariableop_resource:	D
1policy_mlp_dense_3_matmul_readvariableop_resource:	Z@
2policy_mlp_dense_3_biasadd_readvariableop_resource:ZB
.q_mlp_1_dense_4_matmul_readvariableop_resource:
>
/q_mlp_1_dense_4_biasadd_readvariableop_resource:	B
.q_mlp_1_dense_5_matmul_readvariableop_resource:
>
/q_mlp_1_dense_5_biasadd_readvariableop_resource:	B
.q_mlp_1_dense_6_matmul_readvariableop_resource:
>
/q_mlp_1_dense_6_biasadd_readvariableop_resource:	A
.q_mlp_1_dense_7_matmul_readvariableop_resource:	=
/q_mlp_1_dense_7_biasadd_readvariableop_resource:D
0q_1_mlp_2_dense_8_matmul_readvariableop_resource:
@
1q_1_mlp_2_dense_8_biasadd_readvariableop_resource:	D
0q_1_mlp_2_dense_9_matmul_readvariableop_resource:
@
1q_1_mlp_2_dense_9_biasadd_readvariableop_resource:	E
1q_1_mlp_2_dense_10_matmul_readvariableop_resource:
A
2q_1_mlp_2_dense_10_biasadd_readvariableop_resource:	D
1q_1_mlp_2_dense_11_matmul_readvariableop_resource:	@
2q_1_mlp_2_dense_11_biasadd_readvariableop_resource:
identity

identity_1

identity_2

identity_3’'policy/mlp/dense/BiasAdd/ReadVariableOp’&policy/mlp/dense/MatMul/ReadVariableOp’)policy/mlp/dense_1/BiasAdd/ReadVariableOp’(policy/mlp/dense_1/MatMul/ReadVariableOp’)policy/mlp/dense_2/BiasAdd/ReadVariableOp’(policy/mlp/dense_2/MatMul/ReadVariableOp’)policy/mlp/dense_3/BiasAdd/ReadVariableOp’(policy/mlp/dense_3/MatMul/ReadVariableOp’&q/mlp_1/dense_4/BiasAdd/ReadVariableOp’(q/mlp_1/dense_4/BiasAdd_1/ReadVariableOp’%q/mlp_1/dense_4/MatMul/ReadVariableOp’'q/mlp_1/dense_4/MatMul_1/ReadVariableOp’&q/mlp_1/dense_5/BiasAdd/ReadVariableOp’(q/mlp_1/dense_5/BiasAdd_1/ReadVariableOp’%q/mlp_1/dense_5/MatMul/ReadVariableOp’'q/mlp_1/dense_5/MatMul_1/ReadVariableOp’&q/mlp_1/dense_6/BiasAdd/ReadVariableOp’(q/mlp_1/dense_6/BiasAdd_1/ReadVariableOp’%q/mlp_1/dense_6/MatMul/ReadVariableOp’'q/mlp_1/dense_6/MatMul_1/ReadVariableOp’&q/mlp_1/dense_7/BiasAdd/ReadVariableOp’(q/mlp_1/dense_7/BiasAdd_1/ReadVariableOp’%q/mlp_1/dense_7/MatMul/ReadVariableOp’'q/mlp_1/dense_7/MatMul_1/ReadVariableOp’)q_1/mlp_2/dense_10/BiasAdd/ReadVariableOp’(q_1/mlp_2/dense_10/MatMul/ReadVariableOp’)q_1/mlp_2/dense_11/BiasAdd/ReadVariableOp’(q_1/mlp_2/dense_11/MatMul/ReadVariableOp’(q_1/mlp_2/dense_8/BiasAdd/ReadVariableOp’'q_1/mlp_2/dense_8/MatMul/ReadVariableOp’(q_1/mlp_2/dense_9/BiasAdd/ReadVariableOp’'q_1/mlp_2/dense_9/MatMul/ReadVariableOp
&policy/mlp/dense/MatMul/ReadVariableOpReadVariableOp/policy_mlp_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0
policy/mlp/dense/MatMulMatMulinp_0.policy/mlp/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
'policy/mlp/dense/BiasAdd/ReadVariableOpReadVariableOp0policy_mlp_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ͺ
policy/mlp/dense/BiasAddBiasAdd!policy/mlp/dense/MatMul:product:0/policy/mlp/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????s
policy/mlp/dense/SeluSelu!policy/mlp/dense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
(policy/mlp/dense_1/MatMul/ReadVariableOpReadVariableOp1policy_mlp_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0­
policy/mlp/dense_1/MatMulMatMul#policy/mlp/dense/Selu:activations:00policy/mlp/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
)policy/mlp/dense_1/BiasAdd/ReadVariableOpReadVariableOp2policy_mlp_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0°
policy/mlp/dense_1/BiasAddBiasAdd#policy/mlp/dense_1/MatMul:product:01policy/mlp/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????w
policy/mlp/dense_1/SeluSelu#policy/mlp/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
(policy/mlp/dense_2/MatMul/ReadVariableOpReadVariableOp1policy_mlp_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0―
policy/mlp/dense_2/MatMulMatMul%policy/mlp/dense_1/Selu:activations:00policy/mlp/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
)policy/mlp/dense_2/BiasAdd/ReadVariableOpReadVariableOp2policy_mlp_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0°
policy/mlp/dense_2/BiasAddBiasAdd#policy/mlp/dense_2/MatMul:product:01policy/mlp/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????w
policy/mlp/dense_2/SeluSelu#policy/mlp/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
(policy/mlp/dense_3/MatMul/ReadVariableOpReadVariableOp1policy_mlp_dense_3_matmul_readvariableop_resource*
_output_shapes
:	Z*
dtype0?
policy/mlp/dense_3/MatMulMatMul%policy/mlp/dense_2/Selu:activations:00policy/mlp/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z
)policy/mlp/dense_3/BiasAdd/ReadVariableOpReadVariableOp2policy_mlp_dense_3_biasadd_readvariableop_resource*
_output_shapes
:Z*
dtype0―
policy/mlp/dense_3/BiasAddBiasAdd#policy/mlp/dense_3/MatMul:product:01policy/mlp/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z|
policy/mlp/dense_3/SigmoidSigmoid#policy/mlp/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????ZX
q/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????v
q/concatConcatV2inp_0inp_1q/concat/axis:output:0*
N*
T0*(
_output_shapes
:?????????
%q/mlp_1/dense_4/MatMul/ReadVariableOpReadVariableOp.q_mlp_1_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
q/mlp_1/dense_4/MatMulMatMulq/concat:output:0-q/mlp_1/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
&q/mlp_1/dense_4/BiasAdd/ReadVariableOpReadVariableOp/q_mlp_1_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0§
q/mlp_1/dense_4/BiasAddBiasAdd q/mlp_1/dense_4/MatMul:product:0.q/mlp_1/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????q
q/mlp_1/dense_4/SeluSelu q/mlp_1/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
%q/mlp_1/dense_5/MatMul/ReadVariableOpReadVariableOp.q_mlp_1_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0¦
q/mlp_1/dense_5/MatMulMatMul"q/mlp_1/dense_4/Selu:activations:0-q/mlp_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
&q/mlp_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp/q_mlp_1_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0§
q/mlp_1/dense_5/BiasAddBiasAdd q/mlp_1/dense_5/MatMul:product:0.q/mlp_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????q
q/mlp_1/dense_5/SeluSelu q/mlp_1/dense_5/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
%q/mlp_1/dense_6/MatMul/ReadVariableOpReadVariableOp.q_mlp_1_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0¦
q/mlp_1/dense_6/MatMulMatMul"q/mlp_1/dense_5/Selu:activations:0-q/mlp_1/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
&q/mlp_1/dense_6/BiasAdd/ReadVariableOpReadVariableOp/q_mlp_1_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0§
q/mlp_1/dense_6/BiasAddBiasAdd q/mlp_1/dense_6/MatMul:product:0.q/mlp_1/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????q
q/mlp_1/dense_6/SeluSelu q/mlp_1/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
%q/mlp_1/dense_7/MatMul/ReadVariableOpReadVariableOp.q_mlp_1_dense_7_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0₯
q/mlp_1/dense_7/MatMulMatMul"q/mlp_1/dense_6/Selu:activations:0-q/mlp_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
&q/mlp_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp/q_mlp_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¦
q/mlp_1/dense_7/BiasAddBiasAdd q/mlp_1/dense_7/MatMul:product:0.q/mlp_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z
q_1/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????z

q_1/concatConcatV2inp_0inp_1q_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:?????????
'q_1/mlp_2/dense_8/MatMul/ReadVariableOpReadVariableOp0q_1_mlp_2_dense_8_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
q_1/mlp_2/dense_8/MatMulMatMulq_1/concat:output:0/q_1/mlp_2/dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
(q_1/mlp_2/dense_8/BiasAdd/ReadVariableOpReadVariableOp1q_1_mlp_2_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
q_1/mlp_2/dense_8/BiasAddBiasAdd"q_1/mlp_2/dense_8/MatMul:product:00q_1/mlp_2/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????u
q_1/mlp_2/dense_8/SeluSelu"q_1/mlp_2/dense_8/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
'q_1/mlp_2/dense_9/MatMul/ReadVariableOpReadVariableOp0q_1_mlp_2_dense_9_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0¬
q_1/mlp_2/dense_9/MatMulMatMul$q_1/mlp_2/dense_8/Selu:activations:0/q_1/mlp_2/dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
(q_1/mlp_2/dense_9/BiasAdd/ReadVariableOpReadVariableOp1q_1_mlp_2_dense_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
q_1/mlp_2/dense_9/BiasAddBiasAdd"q_1/mlp_2/dense_9/MatMul:product:00q_1/mlp_2/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????u
q_1/mlp_2/dense_9/SeluSelu"q_1/mlp_2/dense_9/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
(q_1/mlp_2/dense_10/MatMul/ReadVariableOpReadVariableOp1q_1_mlp_2_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0?
q_1/mlp_2/dense_10/MatMulMatMul$q_1/mlp_2/dense_9/Selu:activations:00q_1/mlp_2/dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
)q_1/mlp_2/dense_10/BiasAdd/ReadVariableOpReadVariableOp2q_1_mlp_2_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0°
q_1/mlp_2/dense_10/BiasAddBiasAdd#q_1/mlp_2/dense_10/MatMul:product:01q_1/mlp_2/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????w
q_1/mlp_2/dense_10/SeluSelu#q_1/mlp_2/dense_10/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
(q_1/mlp_2/dense_11/MatMul/ReadVariableOpReadVariableOp1q_1_mlp_2_dense_11_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0?
q_1/mlp_2/dense_11/MatMulMatMul%q_1/mlp_2/dense_10/Selu:activations:00q_1/mlp_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
)q_1/mlp_2/dense_11/BiasAdd/ReadVariableOpReadVariableOp2q_1_mlp_2_dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0―
q_1/mlp_2/dense_11/BiasAddBiasAdd#q_1/mlp_2/dense_11/MatMul:product:01q_1/mlp_2/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z
q/concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????

q/concat_1ConcatV2inp_0policy/mlp/dense_3/Sigmoid:y:0q/concat_1/axis:output:0*
N*
T0*(
_output_shapes
:?????????
'q/mlp_1/dense_4/MatMul_1/ReadVariableOpReadVariableOp.q_mlp_1_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
q/mlp_1/dense_4/MatMul_1MatMulq/concat_1:output:0/q/mlp_1/dense_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
(q/mlp_1/dense_4/BiasAdd_1/ReadVariableOpReadVariableOp/q_mlp_1_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
q/mlp_1/dense_4/BiasAdd_1BiasAdd"q/mlp_1/dense_4/MatMul_1:product:00q/mlp_1/dense_4/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????u
q/mlp_1/dense_4/Selu_1Selu"q/mlp_1/dense_4/BiasAdd_1:output:0*
T0*(
_output_shapes
:?????????
'q/mlp_1/dense_5/MatMul_1/ReadVariableOpReadVariableOp.q_mlp_1_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0¬
q/mlp_1/dense_5/MatMul_1MatMul$q/mlp_1/dense_4/Selu_1:activations:0/q/mlp_1/dense_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
(q/mlp_1/dense_5/BiasAdd_1/ReadVariableOpReadVariableOp/q_mlp_1_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
q/mlp_1/dense_5/BiasAdd_1BiasAdd"q/mlp_1/dense_5/MatMul_1:product:00q/mlp_1/dense_5/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????u
q/mlp_1/dense_5/Selu_1Selu"q/mlp_1/dense_5/BiasAdd_1:output:0*
T0*(
_output_shapes
:?????????
'q/mlp_1/dense_6/MatMul_1/ReadVariableOpReadVariableOp.q_mlp_1_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0¬
q/mlp_1/dense_6/MatMul_1MatMul$q/mlp_1/dense_5/Selu_1:activations:0/q/mlp_1/dense_6/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
(q/mlp_1/dense_6/BiasAdd_1/ReadVariableOpReadVariableOp/q_mlp_1_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
q/mlp_1/dense_6/BiasAdd_1BiasAdd"q/mlp_1/dense_6/MatMul_1:product:00q/mlp_1/dense_6/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????u
q/mlp_1/dense_6/Selu_1Selu"q/mlp_1/dense_6/BiasAdd_1:output:0*
T0*(
_output_shapes
:?????????
'q/mlp_1/dense_7/MatMul_1/ReadVariableOpReadVariableOp.q_mlp_1_dense_7_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0«
q/mlp_1/dense_7/MatMul_1MatMul$q/mlp_1/dense_6/Selu_1:activations:0/q/mlp_1/dense_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
(q/mlp_1/dense_7/BiasAdd_1/ReadVariableOpReadVariableOp/q_mlp_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¬
q/mlp_1/dense_7/BiasAdd_1BiasAdd"q/mlp_1/dense_7/MatMul_1:product:00q/mlp_1/dense_7/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m
IdentityIdentitypolicy/mlp/dense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????Zq

Identity_1Identity q/mlp_1/dense_7/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????t

Identity_2Identity#q_1/mlp_2/dense_11/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????s

Identity_3Identity"q/mlp_1/dense_7/BiasAdd_1:output:0^NoOp*
T0*'
_output_shapes
:?????????
NoOpNoOp(^policy/mlp/dense/BiasAdd/ReadVariableOp'^policy/mlp/dense/MatMul/ReadVariableOp*^policy/mlp/dense_1/BiasAdd/ReadVariableOp)^policy/mlp/dense_1/MatMul/ReadVariableOp*^policy/mlp/dense_2/BiasAdd/ReadVariableOp)^policy/mlp/dense_2/MatMul/ReadVariableOp*^policy/mlp/dense_3/BiasAdd/ReadVariableOp)^policy/mlp/dense_3/MatMul/ReadVariableOp'^q/mlp_1/dense_4/BiasAdd/ReadVariableOp)^q/mlp_1/dense_4/BiasAdd_1/ReadVariableOp&^q/mlp_1/dense_4/MatMul/ReadVariableOp(^q/mlp_1/dense_4/MatMul_1/ReadVariableOp'^q/mlp_1/dense_5/BiasAdd/ReadVariableOp)^q/mlp_1/dense_5/BiasAdd_1/ReadVariableOp&^q/mlp_1/dense_5/MatMul/ReadVariableOp(^q/mlp_1/dense_5/MatMul_1/ReadVariableOp'^q/mlp_1/dense_6/BiasAdd/ReadVariableOp)^q/mlp_1/dense_6/BiasAdd_1/ReadVariableOp&^q/mlp_1/dense_6/MatMul/ReadVariableOp(^q/mlp_1/dense_6/MatMul_1/ReadVariableOp'^q/mlp_1/dense_7/BiasAdd/ReadVariableOp)^q/mlp_1/dense_7/BiasAdd_1/ReadVariableOp&^q/mlp_1/dense_7/MatMul/ReadVariableOp(^q/mlp_1/dense_7/MatMul_1/ReadVariableOp*^q_1/mlp_2/dense_10/BiasAdd/ReadVariableOp)^q_1/mlp_2/dense_10/MatMul/ReadVariableOp*^q_1/mlp_2/dense_11/BiasAdd/ReadVariableOp)^q_1/mlp_2/dense_11/MatMul/ReadVariableOp)^q_1/mlp_2/dense_8/BiasAdd/ReadVariableOp(^q_1/mlp_2/dense_8/MatMul/ReadVariableOp)^q_1/mlp_2/dense_9/BiasAdd/ReadVariableOp(^q_1/mlp_2/dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:??????????:?????????Z: : : : : : : : : : : : : : : : : : : : : : : : 2R
'policy/mlp/dense/BiasAdd/ReadVariableOp'policy/mlp/dense/BiasAdd/ReadVariableOp2P
&policy/mlp/dense/MatMul/ReadVariableOp&policy/mlp/dense/MatMul/ReadVariableOp2V
)policy/mlp/dense_1/BiasAdd/ReadVariableOp)policy/mlp/dense_1/BiasAdd/ReadVariableOp2T
(policy/mlp/dense_1/MatMul/ReadVariableOp(policy/mlp/dense_1/MatMul/ReadVariableOp2V
)policy/mlp/dense_2/BiasAdd/ReadVariableOp)policy/mlp/dense_2/BiasAdd/ReadVariableOp2T
(policy/mlp/dense_2/MatMul/ReadVariableOp(policy/mlp/dense_2/MatMul/ReadVariableOp2V
)policy/mlp/dense_3/BiasAdd/ReadVariableOp)policy/mlp/dense_3/BiasAdd/ReadVariableOp2T
(policy/mlp/dense_3/MatMul/ReadVariableOp(policy/mlp/dense_3/MatMul/ReadVariableOp2P
&q/mlp_1/dense_4/BiasAdd/ReadVariableOp&q/mlp_1/dense_4/BiasAdd/ReadVariableOp2T
(q/mlp_1/dense_4/BiasAdd_1/ReadVariableOp(q/mlp_1/dense_4/BiasAdd_1/ReadVariableOp2N
%q/mlp_1/dense_4/MatMul/ReadVariableOp%q/mlp_1/dense_4/MatMul/ReadVariableOp2R
'q/mlp_1/dense_4/MatMul_1/ReadVariableOp'q/mlp_1/dense_4/MatMul_1/ReadVariableOp2P
&q/mlp_1/dense_5/BiasAdd/ReadVariableOp&q/mlp_1/dense_5/BiasAdd/ReadVariableOp2T
(q/mlp_1/dense_5/BiasAdd_1/ReadVariableOp(q/mlp_1/dense_5/BiasAdd_1/ReadVariableOp2N
%q/mlp_1/dense_5/MatMul/ReadVariableOp%q/mlp_1/dense_5/MatMul/ReadVariableOp2R
'q/mlp_1/dense_5/MatMul_1/ReadVariableOp'q/mlp_1/dense_5/MatMul_1/ReadVariableOp2P
&q/mlp_1/dense_6/BiasAdd/ReadVariableOp&q/mlp_1/dense_6/BiasAdd/ReadVariableOp2T
(q/mlp_1/dense_6/BiasAdd_1/ReadVariableOp(q/mlp_1/dense_6/BiasAdd_1/ReadVariableOp2N
%q/mlp_1/dense_6/MatMul/ReadVariableOp%q/mlp_1/dense_6/MatMul/ReadVariableOp2R
'q/mlp_1/dense_6/MatMul_1/ReadVariableOp'q/mlp_1/dense_6/MatMul_1/ReadVariableOp2P
&q/mlp_1/dense_7/BiasAdd/ReadVariableOp&q/mlp_1/dense_7/BiasAdd/ReadVariableOp2T
(q/mlp_1/dense_7/BiasAdd_1/ReadVariableOp(q/mlp_1/dense_7/BiasAdd_1/ReadVariableOp2N
%q/mlp_1/dense_7/MatMul/ReadVariableOp%q/mlp_1/dense_7/MatMul/ReadVariableOp2R
'q/mlp_1/dense_7/MatMul_1/ReadVariableOp'q/mlp_1/dense_7/MatMul_1/ReadVariableOp2V
)q_1/mlp_2/dense_10/BiasAdd/ReadVariableOp)q_1/mlp_2/dense_10/BiasAdd/ReadVariableOp2T
(q_1/mlp_2/dense_10/MatMul/ReadVariableOp(q_1/mlp_2/dense_10/MatMul/ReadVariableOp2V
)q_1/mlp_2/dense_11/BiasAdd/ReadVariableOp)q_1/mlp_2/dense_11/BiasAdd/ReadVariableOp2T
(q_1/mlp_2/dense_11/MatMul/ReadVariableOp(q_1/mlp_2/dense_11/MatMul/ReadVariableOp2T
(q_1/mlp_2/dense_8/BiasAdd/ReadVariableOp(q_1/mlp_2/dense_8/BiasAdd/ReadVariableOp2R
'q_1/mlp_2/dense_8/MatMul/ReadVariableOp'q_1/mlp_2/dense_8/MatMul/ReadVariableOp2T
(q_1/mlp_2/dense_9/BiasAdd/ReadVariableOp(q_1/mlp_2/dense_9/BiasAdd/ReadVariableOp2R
'q_1/mlp_2/dense_9/MatMul/ReadVariableOp'q_1/mlp_2/dense_9/MatMul/ReadVariableOp:N J
'
_output_shapes
:??????????

_user_specified_nameinp/0:NJ
'
_output_shapes
:?????????Z

_user_specified_nameinp/1
ΐ	
Ώ
*__inference_policy_layer_call_fn_178346154
obs
unknown:	?
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	Z
	unknown_6:Z
identity’StatefulPartitionedCall₯
StatefulPartitionedCallStatefulPartitionedCallobsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_policy_layer_call_and_return_conditional_losses_178344520o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:??????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
'
_output_shapes
:??????????

_user_specified_nameobs
Ώ

Μ
%__inference_q_layer_call_fn_178344846
input_1
input_2
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	
	unknown_6:
identity’StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_q_layer_call_and_return_conditional_losses_178344805o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:??????????:?????????Z: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:??????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????Z
!
_user_specified_name	input_2
 
Ξ
'__inference_signature_wrapper_178346112
input_1
input_2
unknown:	?
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	Z
	unknown_6:Z
	unknown_7:

	unknown_8:	
	unknown_9:


unknown_10:	

unknown_11:


unknown_12:	

unknown_13:	

unknown_14:

unknown_15:


unknown_16:	

unknown_17:


unknown_18:	

unknown_19:


unknown_20:	

unknown_21:	

unknown_22:
identity

identity_1

identity_2

identity_3’StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *`
_output_shapesN
L:?????????Z:?????????:?????????:?????????*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference__wrapped_model_178344324o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Zq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:?????????q

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:??????????:?????????Z: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:??????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????Z
!
_user_specified_name	input_2
Ι#
§
B__inference_mlp_layer_call_and_return_conditional_losses_178344457
obs7
$dense_matmul_readvariableop_resource:	?4
%dense_biasadd_readvariableop_resource:	:
&dense_1_matmul_readvariableop_resource:
6
'dense_1_biasadd_readvariableop_resource:	:
&dense_2_matmul_readvariableop_resource:
6
'dense_2_biasadd_readvariableop_resource:	9
&dense_3_matmul_readvariableop_resource:	Z5
'dense_3_biasadd_readvariableop_resource:Z
identity’dense/BiasAdd/ReadVariableOp’dense/MatMul/ReadVariableOp’dense_1/BiasAdd/ReadVariableOp’dense_1/MatMul/ReadVariableOp’dense_2/BiasAdd/ReadVariableOp’dense_2/MatMul/ReadVariableOp’dense_3/BiasAdd/ReadVariableOp’dense_3/MatMul/ReadVariableOp
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0s
dense/MatMulMatMulobs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????]

dense/SeluSeludense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_1/MatMulMatMuldense/Selu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????a
dense_1/SeluSeludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_2/MatMulMatMuldense_1/Selu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????a
dense_2/SeluSeludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	Z*
dtype0
dense_3/MatMulMatMuldense_2/Selu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:Z*
dtype0
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Zf
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????Zb
IdentityIdentitydense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????ZΖ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:??????????: : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:L H
'
_output_shapes
:??????????

_user_specified_nameobs
β)

@__inference_q_layer_call_and_return_conditional_losses_178346296	
inp_0	
inp_1@
,mlp_1_dense_4_matmul_readvariableop_resource:
<
-mlp_1_dense_4_biasadd_readvariableop_resource:	@
,mlp_1_dense_5_matmul_readvariableop_resource:
<
-mlp_1_dense_5_biasadd_readvariableop_resource:	@
,mlp_1_dense_6_matmul_readvariableop_resource:
<
-mlp_1_dense_6_biasadd_readvariableop_resource:	?
,mlp_1_dense_7_matmul_readvariableop_resource:	;
-mlp_1_dense_7_biasadd_readvariableop_resource:
identity’$mlp_1/dense_4/BiasAdd/ReadVariableOp’#mlp_1/dense_4/MatMul/ReadVariableOp’$mlp_1/dense_5/BiasAdd/ReadVariableOp’#mlp_1/dense_5/MatMul/ReadVariableOp’$mlp_1/dense_6/BiasAdd/ReadVariableOp’#mlp_1/dense_6/MatMul/ReadVariableOp’$mlp_1/dense_7/BiasAdd/ReadVariableOp’#mlp_1/dense_7/MatMul/ReadVariableOpV
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????r
concatConcatV2inp_0inp_1concat/axis:output:0*
N*
T0*(
_output_shapes
:?????????
#mlp_1/dense_4/MatMul/ReadVariableOpReadVariableOp,mlp_1_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
mlp_1/dense_4/MatMulMatMulconcat:output:0+mlp_1/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
$mlp_1/dense_4/BiasAdd/ReadVariableOpReadVariableOp-mlp_1_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0‘
mlp_1/dense_4/BiasAddBiasAddmlp_1/dense_4/MatMul:product:0,mlp_1/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????m
mlp_1/dense_4/SeluSelumlp_1/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
#mlp_1/dense_5/MatMul/ReadVariableOpReadVariableOp,mlp_1_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0 
mlp_1/dense_5/MatMulMatMul mlp_1/dense_4/Selu:activations:0+mlp_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
$mlp_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp-mlp_1_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0‘
mlp_1/dense_5/BiasAddBiasAddmlp_1/dense_5/MatMul:product:0,mlp_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????m
mlp_1/dense_5/SeluSelumlp_1/dense_5/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
#mlp_1/dense_6/MatMul/ReadVariableOpReadVariableOp,mlp_1_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0 
mlp_1/dense_6/MatMulMatMul mlp_1/dense_5/Selu:activations:0+mlp_1/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
$mlp_1/dense_6/BiasAdd/ReadVariableOpReadVariableOp-mlp_1_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0‘
mlp_1/dense_6/BiasAddBiasAddmlp_1/dense_6/MatMul:product:0,mlp_1/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????m
mlp_1/dense_6/SeluSelumlp_1/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
#mlp_1/dense_7/MatMul/ReadVariableOpReadVariableOp,mlp_1_dense_7_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
mlp_1/dense_7/MatMulMatMul mlp_1/dense_6/Selu:activations:0+mlp_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
$mlp_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp-mlp_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0 
mlp_1/dense_7/BiasAddBiasAddmlp_1/dense_7/MatMul:product:0,mlp_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m
IdentityIdentitymlp_1/dense_7/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????ϊ
NoOpNoOp%^mlp_1/dense_4/BiasAdd/ReadVariableOp$^mlp_1/dense_4/MatMul/ReadVariableOp%^mlp_1/dense_5/BiasAdd/ReadVariableOp$^mlp_1/dense_5/MatMul/ReadVariableOp%^mlp_1/dense_6/BiasAdd/ReadVariableOp$^mlp_1/dense_6/MatMul/ReadVariableOp%^mlp_1/dense_7/BiasAdd/ReadVariableOp$^mlp_1/dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:??????????:?????????Z: : : : : : : : 2L
$mlp_1/dense_4/BiasAdd/ReadVariableOp$mlp_1/dense_4/BiasAdd/ReadVariableOp2J
#mlp_1/dense_4/MatMul/ReadVariableOp#mlp_1/dense_4/MatMul/ReadVariableOp2L
$mlp_1/dense_5/BiasAdd/ReadVariableOp$mlp_1/dense_5/BiasAdd/ReadVariableOp2J
#mlp_1/dense_5/MatMul/ReadVariableOp#mlp_1/dense_5/MatMul/ReadVariableOp2L
$mlp_1/dense_6/BiasAdd/ReadVariableOp$mlp_1/dense_6/BiasAdd/ReadVariableOp2J
#mlp_1/dense_6/MatMul/ReadVariableOp#mlp_1/dense_6/MatMul/ReadVariableOp2L
$mlp_1/dense_7/BiasAdd/ReadVariableOp$mlp_1/dense_7/BiasAdd/ReadVariableOp2J
#mlp_1/dense_7/MatMul/ReadVariableOp#mlp_1/dense_7/MatMul/ReadVariableOp:N J
'
_output_shapes
:??????????

_user_specified_nameinp/0:NJ
'
_output_shapes
:?????????Z

_user_specified_nameinp/1
Ά
‘
B__inference_q_1_layer_call_and_return_conditional_losses_178345162
input_1
input_2#
mlp_2_178345144:

mlp_2_178345146:	#
mlp_2_178345148:

mlp_2_178345150:	#
mlp_2_178345152:

mlp_2_178345154:	"
mlp_2_178345156:	
mlp_2_178345158:
identity’mlp_2/StatefulPartitionedCallV
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????v
concatConcatV2input_1input_2concat/axis:output:0*
N*
T0*(
_output_shapes
:?????????θ
mlp_2/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0mlp_2_178345144mlp_2_178345146mlp_2_178345148mlp_2_178345150mlp_2_178345152mlp_2_178345154mlp_2_178345156mlp_2_178345158*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_mlp_2_layer_call_and_return_conditional_losses_178344936u
IdentityIdentity&mlp_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????f
NoOpNoOp^mlp_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:??????????:?????????Z: : : : : : : : 2>
mlp_2/StatefulPartitionedCallmlp_2/StatefulPartitionedCall:P L
'
_output_shapes
:??????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????Z
!
_user_specified_name	input_2
 

@__inference_q_layer_call_and_return_conditional_losses_178344805
inp	
inp_1#
mlp_1_178344787:

mlp_1_178344789:	#
mlp_1_178344791:

mlp_1_178344793:	#
mlp_1_178344795:

mlp_1_178344797:	"
mlp_1_178344799:	
mlp_1_178344801:
identity’mlp_1/StatefulPartitionedCallV
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????p
concatConcatV2inpinp_1concat/axis:output:0*
N*
T0*(
_output_shapes
:?????????θ
mlp_1/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0mlp_1_178344787mlp_1_178344789mlp_1_178344791mlp_1_178344793mlp_1_178344795mlp_1_178344797mlp_1_178344799mlp_1_178344801*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_mlp_1_layer_call_and_return_conditional_losses_178344737u
IdentityIdentity&mlp_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????f
NoOpNoOp^mlp_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:??????????:?????????Z: : : : : : : : 2>
mlp_1/StatefulPartitionedCallmlp_1/StatefulPartitionedCall:L H
'
_output_shapes
:??????????

_user_specified_nameinp:LH
'
_output_shapes
:?????????Z

_user_specified_nameinp
’

B__inference_q_1_layer_call_and_return_conditional_losses_178344955
inp	
inp_1#
mlp_2_178344937:

mlp_2_178344939:	#
mlp_2_178344941:

mlp_2_178344943:	#
mlp_2_178344945:

mlp_2_178344947:	"
mlp_2_178344949:	
mlp_2_178344951:
identity’mlp_2/StatefulPartitionedCallV
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????p
concatConcatV2inpinp_1concat/axis:output:0*
N*
T0*(
_output_shapes
:?????????θ
mlp_2/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0mlp_2_178344937mlp_2_178344939mlp_2_178344941mlp_2_178344943mlp_2_178344945mlp_2_178344947mlp_2_178344949mlp_2_178344951*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_mlp_2_layer_call_and_return_conditional_losses_178344936u
IdentityIdentity&mlp_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????f
NoOpNoOp^mlp_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:??????????:?????????Z: : : : : : : : 2>
mlp_2/StatefulPartitionedCallmlp_2/StatefulPartitionedCall:L H
'
_output_shapes
:??????????

_user_specified_nameinp:LH
'
_output_shapes
:?????????Z

_user_specified_nameinp
Γ#
Ί
D__inference_mlp_2_layer_call_and_return_conditional_losses_178345029
obs:
&dense_8_matmul_readvariableop_resource:
6
'dense_8_biasadd_readvariableop_resource:	:
&dense_9_matmul_readvariableop_resource:
6
'dense_9_biasadd_readvariableop_resource:	;
'dense_10_matmul_readvariableop_resource:
7
(dense_10_biasadd_readvariableop_resource:	:
'dense_11_matmul_readvariableop_resource:	6
(dense_11_biasadd_readvariableop_resource:
identity’dense_10/BiasAdd/ReadVariableOp’dense_10/MatMul/ReadVariableOp’dense_11/BiasAdd/ReadVariableOp’dense_11/MatMul/ReadVariableOp’dense_8/BiasAdd/ReadVariableOp’dense_8/MatMul/ReadVariableOp’dense_9/BiasAdd/ReadVariableOp’dense_9/MatMul/ReadVariableOp
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0w
dense_8/MatMulMatMulobs%dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????a
dense_8/SeluSeludense_8/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_9/MatMulMatMuldense_8/Selu:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????a
dense_9/SeluSeludense_9/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_10/MatMulMatMuldense_9/Selu:activations:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????c
dense_10/SeluSeludense_10/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_11/MatMulMatMuldense_10/Selu:activations:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_11/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????Ξ
NoOpNoOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????: : : : : : : : 2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:M I
(
_output_shapes
:?????????

_user_specified_nameobs
Γ#
Ί
D__inference_mlp_2_layer_call_and_return_conditional_losses_178346725
obs:
&dense_8_matmul_readvariableop_resource:
6
'dense_8_biasadd_readvariableop_resource:	:
&dense_9_matmul_readvariableop_resource:
6
'dense_9_biasadd_readvariableop_resource:	;
'dense_10_matmul_readvariableop_resource:
7
(dense_10_biasadd_readvariableop_resource:	:
'dense_11_matmul_readvariableop_resource:	6
(dense_11_biasadd_readvariableop_resource:
identity’dense_10/BiasAdd/ReadVariableOp’dense_10/MatMul/ReadVariableOp’dense_11/BiasAdd/ReadVariableOp’dense_11/MatMul/ReadVariableOp’dense_8/BiasAdd/ReadVariableOp’dense_8/MatMul/ReadVariableOp’dense_9/BiasAdd/ReadVariableOp’dense_9/MatMul/ReadVariableOp
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0w
dense_8/MatMulMatMulobs%dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????a
dense_8/SeluSeludense_8/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_9/MatMulMatMuldense_8/Selu:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????a
dense_9/SeluSeludense_9/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_10/MatMulMatMuldense_9/Selu:activations:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????c
dense_10/SeluSeludense_10/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_11/MatMulMatMuldense_10/Selu:activations:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_11/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????Ξ
NoOpNoOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????: : : : : : : : 2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:M I
(
_output_shapes
:?????????

_user_specified_nameobs
Ζ
Τ
1__inference_mlp_td3_model_layer_call_fn_178345760	
inp_0	
inp_1
unknown:	?
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	Z
	unknown_6:Z
	unknown_7:

	unknown_8:	
	unknown_9:


unknown_10:	

unknown_11:


unknown_12:	

unknown_13:	

unknown_14:

unknown_15:


unknown_16:	

unknown_17:


unknown_18:	

unknown_19:


unknown_20:	

unknown_21:	

unknown_22:
identity

identity_1

identity_2

identity_3’StatefulPartitionedCallΟ
StatefulPartitionedCallStatefulPartitionedCallinp_0inp_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *`
_output_shapesN
L:?????????Z:?????????:?????????:?????????*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_mlp_td3_model_layer_call_and_return_conditional_losses_178345259o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Zq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:?????????q

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:??????????:?????????Z: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:??????????

_user_specified_nameinp/0:NJ
'
_output_shapes
:?????????Z

_user_specified_nameinp/1
τ!
―
L__inference_mlp_td3_model_layer_call_and_return_conditional_losses_178345700
input_1
input_2#
policy_178345636:	?
policy_178345638:	$
policy_178345640:

policy_178345642:	$
policy_178345644:

policy_178345646:	#
policy_178345648:	Z
policy_178345650:Z
q_178345653:

q_178345655:	
q_178345657:

q_178345659:	
q_178345661:

q_178345663:	
q_178345665:	
q_178345667:!
q_1_178345670:

q_1_178345672:	!
q_1_178345674:

q_1_178345676:	!
q_1_178345678:

q_1_178345680:	 
q_1_178345682:	
q_1_178345684:
identity

identity_1

identity_2

identity_3’policy/StatefulPartitionedCall’q/StatefulPartitionedCall’q/StatefulPartitionedCall_1’q_1/StatefulPartitionedCallκ
policy/StatefulPartitionedCallStatefulPartitionedCallinput_1policy_178345636policy_178345638policy_178345640policy_178345642policy_178345644policy_178345646policy_178345648policy_178345650*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_policy_layer_call_and_return_conditional_losses_178344520Β
q/StatefulPartitionedCallStatefulPartitionedCallinput_1input_2q_178345653q_178345655q_178345657q_178345659q_178345661q_178345663q_178345665q_178345667*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_q_layer_call_and_return_conditional_losses_178344805Φ
q_1/StatefulPartitionedCallStatefulPartitionedCallinput_1input_2q_1_178345670q_1_178345672q_1_178345674q_1_178345676q_1_178345678q_1_178345680q_1_178345682q_1_178345684*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_q_1_layer_call_and_return_conditional_losses_178345097δ
q/StatefulPartitionedCall_1StatefulPartitionedCallinput_1'policy/StatefulPartitionedCall:output:0q_178345653q_178345655q_178345657q_178345659q_178345661q_178345663q_178345665q_178345667*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_q_layer_call_and_return_conditional_losses_178344805v
IdentityIdentity'policy/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Zs

Identity_1Identity"q/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????u

Identity_2Identity$q_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????u

Identity_3Identity$q/StatefulPartitionedCall_1:output:0^NoOp*
T0*'
_output_shapes
:?????????Ώ
NoOpNoOp^policy/StatefulPartitionedCall^q/StatefulPartitionedCall^q/StatefulPartitionedCall_1^q_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:??????????:?????????Z: : : : : : : : : : : : : : : : : : : : : : : : 2@
policy/StatefulPartitionedCallpolicy/StatefulPartitionedCall26
q/StatefulPartitionedCallq/StatefulPartitionedCall2:
q/StatefulPartitionedCall_1q/StatefulPartitionedCall_12:
q_1/StatefulPartitionedCallq_1/StatefulPartitionedCall:P L
'
_output_shapes
:??????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????Z
!
_user_specified_name	input_2
Ώ

Μ
%__inference_q_layer_call_fn_178344682
input_1
input_2
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	
	unknown_6:
identity’StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_q_layer_call_and_return_conditional_losses_178344663o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:??????????:?????????Z: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:??????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????Z
!
_user_specified_name	input_2
Α	
Ώ
)__inference_mlp_2_layer_call_fn_178346673
obs
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	
	unknown_6:
identity’StatefulPartitionedCall€
StatefulPartitionedCallStatefulPartitionedCallobsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_mlp_2_layer_call_and_return_conditional_losses_178344936o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:M I
(
_output_shapes
:?????????

_user_specified_nameobs
Α	
Ώ
)__inference_mlp_2_layer_call_fn_178346694
obs
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	
	unknown_6:
identity’StatefulPartitionedCall€
StatefulPartitionedCallStatefulPartitionedCallobsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_mlp_2_layer_call_and_return_conditional_losses_178345029o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:M I
(
_output_shapes
:?????????

_user_specified_nameobs
*
₯
B__inference_q_1_layer_call_and_return_conditional_losses_178346442	
inp_0	
inp_1@
,mlp_2_dense_8_matmul_readvariableop_resource:
<
-mlp_2_dense_8_biasadd_readvariableop_resource:	@
,mlp_2_dense_9_matmul_readvariableop_resource:
<
-mlp_2_dense_9_biasadd_readvariableop_resource:	A
-mlp_2_dense_10_matmul_readvariableop_resource:
=
.mlp_2_dense_10_biasadd_readvariableop_resource:	@
-mlp_2_dense_11_matmul_readvariableop_resource:	<
.mlp_2_dense_11_biasadd_readvariableop_resource:
identity’%mlp_2/dense_10/BiasAdd/ReadVariableOp’$mlp_2/dense_10/MatMul/ReadVariableOp’%mlp_2/dense_11/BiasAdd/ReadVariableOp’$mlp_2/dense_11/MatMul/ReadVariableOp’$mlp_2/dense_8/BiasAdd/ReadVariableOp’#mlp_2/dense_8/MatMul/ReadVariableOp’$mlp_2/dense_9/BiasAdd/ReadVariableOp’#mlp_2/dense_9/MatMul/ReadVariableOpV
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????r
concatConcatV2inp_0inp_1concat/axis:output:0*
N*
T0*(
_output_shapes
:?????????
#mlp_2/dense_8/MatMul/ReadVariableOpReadVariableOp,mlp_2_dense_8_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
mlp_2/dense_8/MatMulMatMulconcat:output:0+mlp_2/dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
$mlp_2/dense_8/BiasAdd/ReadVariableOpReadVariableOp-mlp_2_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0‘
mlp_2/dense_8/BiasAddBiasAddmlp_2/dense_8/MatMul:product:0,mlp_2/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????m
mlp_2/dense_8/SeluSelumlp_2/dense_8/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
#mlp_2/dense_9/MatMul/ReadVariableOpReadVariableOp,mlp_2_dense_9_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0 
mlp_2/dense_9/MatMulMatMul mlp_2/dense_8/Selu:activations:0+mlp_2/dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
$mlp_2/dense_9/BiasAdd/ReadVariableOpReadVariableOp-mlp_2_dense_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0‘
mlp_2/dense_9/BiasAddBiasAddmlp_2/dense_9/MatMul:product:0,mlp_2/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????m
mlp_2/dense_9/SeluSelumlp_2/dense_9/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
$mlp_2/dense_10/MatMul/ReadVariableOpReadVariableOp-mlp_2_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0’
mlp_2/dense_10/MatMulMatMul mlp_2/dense_9/Selu:activations:0,mlp_2/dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
%mlp_2/dense_10/BiasAdd/ReadVariableOpReadVariableOp.mlp_2_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0€
mlp_2/dense_10/BiasAddBiasAddmlp_2/dense_10/MatMul:product:0-mlp_2/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????o
mlp_2/dense_10/SeluSelumlp_2/dense_10/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
$mlp_2/dense_11/MatMul/ReadVariableOpReadVariableOp-mlp_2_dense_11_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0’
mlp_2/dense_11/MatMulMatMul!mlp_2/dense_10/Selu:activations:0,mlp_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
%mlp_2/dense_11/BiasAdd/ReadVariableOpReadVariableOp.mlp_2_dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0£
mlp_2/dense_11/BiasAddBiasAddmlp_2/dense_11/MatMul:product:0-mlp_2/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????n
IdentityIdentitymlp_2/dense_11/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????ώ
NoOpNoOp&^mlp_2/dense_10/BiasAdd/ReadVariableOp%^mlp_2/dense_10/MatMul/ReadVariableOp&^mlp_2/dense_11/BiasAdd/ReadVariableOp%^mlp_2/dense_11/MatMul/ReadVariableOp%^mlp_2/dense_8/BiasAdd/ReadVariableOp$^mlp_2/dense_8/MatMul/ReadVariableOp%^mlp_2/dense_9/BiasAdd/ReadVariableOp$^mlp_2/dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:??????????:?????????Z: : : : : : : : 2N
%mlp_2/dense_10/BiasAdd/ReadVariableOp%mlp_2/dense_10/BiasAdd/ReadVariableOp2L
$mlp_2/dense_10/MatMul/ReadVariableOp$mlp_2/dense_10/MatMul/ReadVariableOp2N
%mlp_2/dense_11/BiasAdd/ReadVariableOp%mlp_2/dense_11/BiasAdd/ReadVariableOp2L
$mlp_2/dense_11/MatMul/ReadVariableOp$mlp_2/dense_11/MatMul/ReadVariableOp2L
$mlp_2/dense_8/BiasAdd/ReadVariableOp$mlp_2/dense_8/BiasAdd/ReadVariableOp2J
#mlp_2/dense_8/MatMul/ReadVariableOp#mlp_2/dense_8/MatMul/ReadVariableOp2L
$mlp_2/dense_9/BiasAdd/ReadVariableOp$mlp_2/dense_9/BiasAdd/ReadVariableOp2J
#mlp_2/dense_9/MatMul/ReadVariableOp#mlp_2/dense_9/MatMul/ReadVariableOp:N J
'
_output_shapes
:??????????

_user_specified_nameinp/0:NJ
'
_output_shapes
:?????????Z

_user_specified_nameinp/1
΄

@__inference_q_layer_call_and_return_conditional_losses_178344894
input_1
input_2#
mlp_1_178344876:

mlp_1_178344878:	#
mlp_1_178344880:

mlp_1_178344882:	#
mlp_1_178344884:

mlp_1_178344886:	"
mlp_1_178344888:	
mlp_1_178344890:
identity’mlp_1/StatefulPartitionedCallV
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????v
concatConcatV2input_1input_2concat/axis:output:0*
N*
T0*(
_output_shapes
:?????????θ
mlp_1/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0mlp_1_178344876mlp_1_178344878mlp_1_178344880mlp_1_178344882mlp_1_178344884mlp_1_178344886mlp_1_178344888mlp_1_178344890*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_mlp_1_layer_call_and_return_conditional_losses_178344737u
IdentityIdentity&mlp_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????f
NoOpNoOp^mlp_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:??????????:?????????Z: : : : : : : : 2>
mlp_1/StatefulPartitionedCallmlp_1/StatefulPartitionedCall:P L
'
_output_shapes
:??????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????Z
!
_user_specified_name	input_2
­&
κ
E__inference_policy_layer_call_and_return_conditional_losses_178346218
obs;
(mlp_dense_matmul_readvariableop_resource:	?8
)mlp_dense_biasadd_readvariableop_resource:	>
*mlp_dense_1_matmul_readvariableop_resource:
:
+mlp_dense_1_biasadd_readvariableop_resource:	>
*mlp_dense_2_matmul_readvariableop_resource:
:
+mlp_dense_2_biasadd_readvariableop_resource:	=
*mlp_dense_3_matmul_readvariableop_resource:	Z9
+mlp_dense_3_biasadd_readvariableop_resource:Z
identity’ mlp/dense/BiasAdd/ReadVariableOp’mlp/dense/MatMul/ReadVariableOp’"mlp/dense_1/BiasAdd/ReadVariableOp’!mlp/dense_1/MatMul/ReadVariableOp’"mlp/dense_2/BiasAdd/ReadVariableOp’!mlp/dense_2/MatMul/ReadVariableOp’"mlp/dense_3/BiasAdd/ReadVariableOp’!mlp/dense_3/MatMul/ReadVariableOp
mlp/dense/MatMul/ReadVariableOpReadVariableOp(mlp_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0{
mlp/dense/MatMulMatMulobs'mlp/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
 mlp/dense/BiasAdd/ReadVariableOpReadVariableOp)mlp_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
mlp/dense/BiasAddBiasAddmlp/dense/MatMul:product:0(mlp/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????e
mlp/dense/SeluSelumlp/dense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
!mlp/dense_1/MatMul/ReadVariableOpReadVariableOp*mlp_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
mlp/dense_1/MatMulMatMulmlp/dense/Selu:activations:0)mlp/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
"mlp/dense_1/BiasAdd/ReadVariableOpReadVariableOp+mlp_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
mlp/dense_1/BiasAddBiasAddmlp/dense_1/MatMul:product:0*mlp/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????i
mlp/dense_1/SeluSelumlp/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
!mlp/dense_2/MatMul/ReadVariableOpReadVariableOp*mlp_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
mlp/dense_2/MatMulMatMulmlp/dense_1/Selu:activations:0)mlp/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
"mlp/dense_2/BiasAdd/ReadVariableOpReadVariableOp+mlp_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
mlp/dense_2/BiasAddBiasAddmlp/dense_2/MatMul:product:0*mlp/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????i
mlp/dense_2/SeluSelumlp/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
!mlp/dense_3/MatMul/ReadVariableOpReadVariableOp*mlp_dense_3_matmul_readvariableop_resource*
_output_shapes
:	Z*
dtype0
mlp/dense_3/MatMulMatMulmlp/dense_2/Selu:activations:0)mlp/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z
"mlp/dense_3/BiasAdd/ReadVariableOpReadVariableOp+mlp_dense_3_biasadd_readvariableop_resource*
_output_shapes
:Z*
dtype0
mlp/dense_3/BiasAddBiasAddmlp/dense_3/MatMul:product:0*mlp/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Zn
mlp/dense_3/SigmoidSigmoidmlp/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????Zf
IdentityIdentitymlp/dense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????Zζ
NoOpNoOp!^mlp/dense/BiasAdd/ReadVariableOp ^mlp/dense/MatMul/ReadVariableOp#^mlp/dense_1/BiasAdd/ReadVariableOp"^mlp/dense_1/MatMul/ReadVariableOp#^mlp/dense_2/BiasAdd/ReadVariableOp"^mlp/dense_2/MatMul/ReadVariableOp#^mlp/dense_3/BiasAdd/ReadVariableOp"^mlp/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:??????????: : : : : : : : 2D
 mlp/dense/BiasAdd/ReadVariableOp mlp/dense/BiasAdd/ReadVariableOp2B
mlp/dense/MatMul/ReadVariableOpmlp/dense/MatMul/ReadVariableOp2H
"mlp/dense_1/BiasAdd/ReadVariableOp"mlp/dense_1/BiasAdd/ReadVariableOp2F
!mlp/dense_1/MatMul/ReadVariableOp!mlp/dense_1/MatMul/ReadVariableOp2H
"mlp/dense_2/BiasAdd/ReadVariableOp"mlp/dense_2/BiasAdd/ReadVariableOp2F
!mlp/dense_2/MatMul/ReadVariableOp!mlp/dense_2/MatMul/ReadVariableOp2H
"mlp/dense_3/BiasAdd/ReadVariableOp"mlp/dense_3/BiasAdd/ReadVariableOp2F
!mlp/dense_3/MatMul/ReadVariableOp!mlp/dense_3/MatMul/ReadVariableOp:L H
'
_output_shapes
:??????????

_user_specified_nameobs
’

B__inference_q_1_layer_call_and_return_conditional_losses_178345097
inp	
inp_1#
mlp_2_178345079:

mlp_2_178345081:	#
mlp_2_178345083:

mlp_2_178345085:	#
mlp_2_178345087:

mlp_2_178345089:	"
mlp_2_178345091:	
mlp_2_178345093:
identity’mlp_2/StatefulPartitionedCallV
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????p
concatConcatV2inpinp_1concat/axis:output:0*
N*
T0*(
_output_shapes
:?????????θ
mlp_2/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0mlp_2_178345079mlp_2_178345081mlp_2_178345083mlp_2_178345085mlp_2_178345087mlp_2_178345089mlp_2_178345091mlp_2_178345093*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_mlp_2_layer_call_and_return_conditional_losses_178345029u
IdentityIdentity&mlp_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????f
NoOpNoOp^mlp_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:??????????:?????????Z: : : : : : : : 2>
mlp_2/StatefulPartitionedCallmlp_2/StatefulPartitionedCall:L H
'
_output_shapes
:??????????

_user_specified_nameinp:LH
'
_output_shapes
:?????????Z

_user_specified_nameinp
?!
©
L__inference_mlp_td3_model_layer_call_and_return_conditional_losses_178345259
inp	
inp_1#
policy_178345195:	?
policy_178345197:	$
policy_178345199:

policy_178345201:	$
policy_178345203:

policy_178345205:	#
policy_178345207:	Z
policy_178345209:Z
q_178345212:

q_178345214:	
q_178345216:

q_178345218:	
q_178345220:

q_178345222:	
q_178345224:	
q_178345226:!
q_1_178345229:

q_1_178345231:	!
q_1_178345233:

q_1_178345235:	!
q_1_178345237:

q_1_178345239:	 
q_1_178345241:	
q_1_178345243:
identity

identity_1

identity_2

identity_3’policy/StatefulPartitionedCall’q/StatefulPartitionedCall’q/StatefulPartitionedCall_1’q_1/StatefulPartitionedCallζ
policy/StatefulPartitionedCallStatefulPartitionedCallinppolicy_178345195policy_178345197policy_178345199policy_178345201policy_178345203policy_178345205policy_178345207policy_178345209*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_policy_layer_call_and_return_conditional_losses_178344382Ό
q/StatefulPartitionedCallStatefulPartitionedCallinpinp_1q_178345212q_178345214q_178345216q_178345218q_178345220q_178345222q_178345224q_178345226*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_q_layer_call_and_return_conditional_losses_178344663Π
q_1/StatefulPartitionedCallStatefulPartitionedCallinpinp_1q_1_178345229q_1_178345231q_1_178345233q_1_178345235q_1_178345237q_1_178345239q_1_178345241q_1_178345243*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_q_1_layer_call_and_return_conditional_losses_178344955ΰ
q/StatefulPartitionedCall_1StatefulPartitionedCallinp'policy/StatefulPartitionedCall:output:0q_178345212q_178345214q_178345216q_178345218q_178345220q_178345222q_178345224q_178345226*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_q_layer_call_and_return_conditional_losses_178344663v
IdentityIdentity'policy/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Zs

Identity_1Identity"q/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????u

Identity_2Identity$q_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????u

Identity_3Identity$q/StatefulPartitionedCall_1:output:0^NoOp*
T0*'
_output_shapes
:?????????Ώ
NoOpNoOp^policy/StatefulPartitionedCall^q/StatefulPartitionedCall^q/StatefulPartitionedCall_1^q_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:??????????:?????????Z: : : : : : : : : : : : : : : : : : : : : : : : 2@
policy/StatefulPartitionedCallpolicy/StatefulPartitionedCall26
q/StatefulPartitionedCallq/StatefulPartitionedCall2:
q/StatefulPartitionedCall_1q/StatefulPartitionedCall_12:
q_1/StatefulPartitionedCallq_1/StatefulPartitionedCall:L H
'
_output_shapes
:??????????

_user_specified_nameinp:LH
'
_output_shapes
:?????????Z

_user_specified_nameinp
Ά
‘
B__inference_q_1_layer_call_and_return_conditional_losses_178345186
input_1
input_2#
mlp_2_178345168:

mlp_2_178345170:	#
mlp_2_178345172:

mlp_2_178345174:	#
mlp_2_178345176:

mlp_2_178345178:	"
mlp_2_178345180:	
mlp_2_178345182:
identity’mlp_2/StatefulPartitionedCallV
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????v
concatConcatV2input_1input_2concat/axis:output:0*
N*
T0*(
_output_shapes
:?????????θ
mlp_2/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0mlp_2_178345168mlp_2_178345170mlp_2_178345172mlp_2_178345174mlp_2_178345176mlp_2_178345178mlp_2_178345180mlp_2_178345182*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_mlp_2_layer_call_and_return_conditional_losses_178345029u
IdentityIdentity&mlp_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????f
NoOpNoOp^mlp_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:??????????:?????????Z: : : : : : : : 2>
mlp_2/StatefulPartitionedCallmlp_2/StatefulPartitionedCall:P L
'
_output_shapes
:??????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????Z
!
_user_specified_name	input_2
#
²
D__inference_mlp_1_layer_call_and_return_conditional_losses_178346652
obs:
&dense_4_matmul_readvariableop_resource:
6
'dense_4_biasadd_readvariableop_resource:	:
&dense_5_matmul_readvariableop_resource:
6
'dense_5_biasadd_readvariableop_resource:	:
&dense_6_matmul_readvariableop_resource:
6
'dense_6_biasadd_readvariableop_resource:	9
&dense_7_matmul_readvariableop_resource:	5
'dense_7_biasadd_readvariableop_resource:
identity’dense_4/BiasAdd/ReadVariableOp’dense_4/MatMul/ReadVariableOp’dense_5/BiasAdd/ReadVariableOp’dense_5/MatMul/ReadVariableOp’dense_6/BiasAdd/ReadVariableOp’dense_6/MatMul/ReadVariableOp’dense_7/BiasAdd/ReadVariableOp’dense_7/MatMul/ReadVariableOp
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0w
dense_4/MatMulMatMulobs%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????a
dense_4/SeluSeludense_4/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_5/MatMulMatMuldense_4/Selu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????a
dense_5/SeluSeludense_5/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_6/MatMulMatMuldense_5/Selu:activations:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????a
dense_6/SeluSeludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_7/MatMulMatMuldense_6/Selu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_7/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????Κ
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????: : : : : : : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:M I
(
_output_shapes
:?????????

_user_specified_nameobs
΄

@__inference_q_layer_call_and_return_conditional_losses_178344870
input_1
input_2#
mlp_1_178344852:

mlp_1_178344854:	#
mlp_1_178344856:

mlp_1_178344858:	#
mlp_1_178344860:

mlp_1_178344862:	"
mlp_1_178344864:	
mlp_1_178344866:
identity’mlp_1/StatefulPartitionedCallV
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????v
concatConcatV2input_1input_2concat/axis:output:0*
N*
T0*(
_output_shapes
:?????????θ
mlp_1/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0mlp_1_178344852mlp_1_178344854mlp_1_178344856mlp_1_178344858mlp_1_178344860mlp_1_178344862mlp_1_178344864mlp_1_178344866*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_mlp_1_layer_call_and_return_conditional_losses_178344644u
IdentityIdentity&mlp_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????f
NoOpNoOp^mlp_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:??????????:?????????Z: : : : : : : : 2>
mlp_1/StatefulPartitionedCallmlp_1/StatefulPartitionedCall:P L
'
_output_shapes
:??????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????Z
!
_user_specified_name	input_2
ΐ


E__inference_policy_layer_call_and_return_conditional_losses_178344581
input_1 
mlp_178344563:	?
mlp_178344565:	!
mlp_178344567:

mlp_178344569:	!
mlp_178344571:

mlp_178344573:	 
mlp_178344575:	Z
mlp_178344577:Z
identity’mlp/StatefulPartitionedCallΜ
mlp/StatefulPartitionedCallStatefulPartitionedCallinput_1mlp_178344563mlp_178344565mlp_178344567mlp_178344569mlp_178344571mlp_178344573mlp_178344575mlp_178344577*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_mlp_layer_call_and_return_conditional_losses_178344363s
IdentityIdentity$mlp/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Zd
NoOpNoOp^mlp/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:??????????: : : : : : : : 2:
mlp/StatefulPartitionedCallmlp/StatefulPartitionedCall:P L
'
_output_shapes
:??????????
!
_user_specified_name	input_1
 

@__inference_q_layer_call_and_return_conditional_losses_178344663
inp	
inp_1#
mlp_1_178344645:

mlp_1_178344647:	#
mlp_1_178344649:

mlp_1_178344651:	#
mlp_1_178344653:

mlp_1_178344655:	"
mlp_1_178344657:	
mlp_1_178344659:
identity’mlp_1/StatefulPartitionedCallV
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????p
concatConcatV2inpinp_1concat/axis:output:0*
N*
T0*(
_output_shapes
:?????????θ
mlp_1/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0mlp_1_178344645mlp_1_178344647mlp_1_178344649mlp_1_178344651mlp_1_178344653mlp_1_178344655mlp_1_178344657mlp_1_178344659*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_mlp_1_layer_call_and_return_conditional_losses_178344644u
IdentityIdentity&mlp_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????f
NoOpNoOp^mlp_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:??????????:?????????Z: : : : : : : : 2>
mlp_1/StatefulPartitionedCallmlp_1/StatefulPartitionedCall:L H
'
_output_shapes
:??????????

_user_specified_nameinp:LH
'
_output_shapes
:?????????Z

_user_specified_nameinp
τ!
―
L__inference_mlp_td3_model_layer_call_and_return_conditional_losses_178345632
input_1
input_2#
policy_178345568:	?
policy_178345570:	$
policy_178345572:

policy_178345574:	$
policy_178345576:

policy_178345578:	#
policy_178345580:	Z
policy_178345582:Z
q_178345585:

q_178345587:	
q_178345589:

q_178345591:	
q_178345593:

q_178345595:	
q_178345597:	
q_178345599:!
q_1_178345602:

q_1_178345604:	!
q_1_178345606:

q_1_178345608:	!
q_1_178345610:

q_1_178345612:	 
q_1_178345614:	
q_1_178345616:
identity

identity_1

identity_2

identity_3’policy/StatefulPartitionedCall’q/StatefulPartitionedCall’q/StatefulPartitionedCall_1’q_1/StatefulPartitionedCallκ
policy/StatefulPartitionedCallStatefulPartitionedCallinput_1policy_178345568policy_178345570policy_178345572policy_178345574policy_178345576policy_178345578policy_178345580policy_178345582*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_policy_layer_call_and_return_conditional_losses_178344382Β
q/StatefulPartitionedCallStatefulPartitionedCallinput_1input_2q_178345585q_178345587q_178345589q_178345591q_178345593q_178345595q_178345597q_178345599*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_q_layer_call_and_return_conditional_losses_178344663Φ
q_1/StatefulPartitionedCallStatefulPartitionedCallinput_1input_2q_1_178345602q_1_178345604q_1_178345606q_1_178345608q_1_178345610q_1_178345612q_1_178345614q_1_178345616*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_q_1_layer_call_and_return_conditional_losses_178344955δ
q/StatefulPartitionedCall_1StatefulPartitionedCallinput_1'policy/StatefulPartitionedCall:output:0q_178345585q_178345587q_178345589q_178345591q_178345593q_178345595q_178345597q_178345599*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_q_layer_call_and_return_conditional_losses_178344663v
IdentityIdentity'policy/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Zs

Identity_1Identity"q/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????u

Identity_2Identity$q_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????u

Identity_3Identity$q/StatefulPartitionedCall_1:output:0^NoOp*
T0*'
_output_shapes
:?????????Ώ
NoOpNoOp^policy/StatefulPartitionedCall^q/StatefulPartitionedCall^q/StatefulPartitionedCall_1^q_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:??????????:?????????Z: : : : : : : : : : : : : : : : : : : : : : : : 2@
policy/StatefulPartitionedCallpolicy/StatefulPartitionedCall26
q/StatefulPartitionedCallq/StatefulPartitionedCall2:
q/StatefulPartitionedCall_1q/StatefulPartitionedCall_12:
q_1/StatefulPartitionedCallq_1/StatefulPartitionedCall:P L
'
_output_shapes
:??????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????Z
!
_user_specified_name	input_2
#
²
D__inference_mlp_1_layer_call_and_return_conditional_losses_178344644
obs:
&dense_4_matmul_readvariableop_resource:
6
'dense_4_biasadd_readvariableop_resource:	:
&dense_5_matmul_readvariableop_resource:
6
'dense_5_biasadd_readvariableop_resource:	:
&dense_6_matmul_readvariableop_resource:
6
'dense_6_biasadd_readvariableop_resource:	9
&dense_7_matmul_readvariableop_resource:	5
'dense_7_biasadd_readvariableop_resource:
identity’dense_4/BiasAdd/ReadVariableOp’dense_4/MatMul/ReadVariableOp’dense_5/BiasAdd/ReadVariableOp’dense_5/MatMul/ReadVariableOp’dense_6/BiasAdd/ReadVariableOp’dense_6/MatMul/ReadVariableOp’dense_7/BiasAdd/ReadVariableOp’dense_7/MatMul/ReadVariableOp
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0w
dense_4/MatMulMatMulobs%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????a
dense_4/SeluSeludense_4/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_5/MatMulMatMuldense_4/Selu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????a
dense_5/SeluSeludense_5/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_6/MatMulMatMuldense_5/Selu:activations:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????a
dense_6/SeluSeludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_7/MatMulMatMuldense_6/Selu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_7/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????Κ
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????: : : : : : : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:M I
(
_output_shapes
:?????????

_user_specified_nameobs
?
Ψ
1__inference_mlp_td3_model_layer_call_fn_178345564
input_1
input_2
unknown:	?
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	Z
	unknown_6:Z
	unknown_7:

	unknown_8:	
	unknown_9:


unknown_10:	

unknown_11:


unknown_12:	

unknown_13:	

unknown_14:

unknown_15:


unknown_16:	

unknown_17:


unknown_18:	

unknown_19:


unknown_20:	

unknown_21:	

unknown_22:
identity

identity_1

identity_2

identity_3’StatefulPartitionedCallΣ
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *`
_output_shapesN
L:?????????Z:?????????:?????????:?????????*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_mlp_td3_model_layer_call_and_return_conditional_losses_178345447o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Zq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:?????????q

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:??????????:?????????Z: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:??????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????Z
!
_user_specified_name	input_2
Α	
Ώ
)__inference_mlp_1_layer_call_fn_178346590
obs
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	
	unknown_6:
identity’StatefulPartitionedCall€
StatefulPartitionedCallStatefulPartitionedCallobsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_mlp_1_layer_call_and_return_conditional_losses_178344737o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:M I
(
_output_shapes
:?????????

_user_specified_nameobs
Ή;
χ
"__inference__traced_save_178346855
file_prefixD
@savev2_mlp_td3_model_policy_mlp_dense_kernel_read_readvariableopB
>savev2_mlp_td3_model_policy_mlp_dense_bias_read_readvariableopF
Bsavev2_mlp_td3_model_policy_mlp_dense_1_kernel_read_readvariableopD
@savev2_mlp_td3_model_policy_mlp_dense_1_bias_read_readvariableopF
Bsavev2_mlp_td3_model_policy_mlp_dense_2_kernel_read_readvariableopD
@savev2_mlp_td3_model_policy_mlp_dense_2_bias_read_readvariableopF
Bsavev2_mlp_td3_model_policy_mlp_dense_3_kernel_read_readvariableopD
@savev2_mlp_td3_model_policy_mlp_dense_3_bias_read_readvariableopC
?savev2_mlp_td3_model_q_mlp_1_dense_4_kernel_read_readvariableopA
=savev2_mlp_td3_model_q_mlp_1_dense_4_bias_read_readvariableopC
?savev2_mlp_td3_model_q_mlp_1_dense_5_kernel_read_readvariableopA
=savev2_mlp_td3_model_q_mlp_1_dense_5_bias_read_readvariableopC
?savev2_mlp_td3_model_q_mlp_1_dense_6_kernel_read_readvariableopA
=savev2_mlp_td3_model_q_mlp_1_dense_6_bias_read_readvariableopC
?savev2_mlp_td3_model_q_mlp_1_dense_7_kernel_read_readvariableopA
=savev2_mlp_td3_model_q_mlp_1_dense_7_bias_read_readvariableopE
Asavev2_mlp_td3_model_q_1_mlp_2_dense_8_kernel_read_readvariableopC
?savev2_mlp_td3_model_q_1_mlp_2_dense_8_bias_read_readvariableopE
Asavev2_mlp_td3_model_q_1_mlp_2_dense_9_kernel_read_readvariableopC
?savev2_mlp_td3_model_q_1_mlp_2_dense_9_bias_read_readvariableopF
Bsavev2_mlp_td3_model_q_1_mlp_2_dense_10_kernel_read_readvariableopD
@savev2_mlp_td3_model_q_1_mlp_2_dense_10_bias_read_readvariableopF
Bsavev2_mlp_td3_model_q_1_mlp_2_dense_11_kernel_read_readvariableopD
@savev2_mlp_td3_model_q_1_mlp_2_dense_11_bias_read_readvariableop
savev2_const

identity_1’MergeV2Checkpointsw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ψ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueχBτB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B π
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0@savev2_mlp_td3_model_policy_mlp_dense_kernel_read_readvariableop>savev2_mlp_td3_model_policy_mlp_dense_bias_read_readvariableopBsavev2_mlp_td3_model_policy_mlp_dense_1_kernel_read_readvariableop@savev2_mlp_td3_model_policy_mlp_dense_1_bias_read_readvariableopBsavev2_mlp_td3_model_policy_mlp_dense_2_kernel_read_readvariableop@savev2_mlp_td3_model_policy_mlp_dense_2_bias_read_readvariableopBsavev2_mlp_td3_model_policy_mlp_dense_3_kernel_read_readvariableop@savev2_mlp_td3_model_policy_mlp_dense_3_bias_read_readvariableop?savev2_mlp_td3_model_q_mlp_1_dense_4_kernel_read_readvariableop=savev2_mlp_td3_model_q_mlp_1_dense_4_bias_read_readvariableop?savev2_mlp_td3_model_q_mlp_1_dense_5_kernel_read_readvariableop=savev2_mlp_td3_model_q_mlp_1_dense_5_bias_read_readvariableop?savev2_mlp_td3_model_q_mlp_1_dense_6_kernel_read_readvariableop=savev2_mlp_td3_model_q_mlp_1_dense_6_bias_read_readvariableop?savev2_mlp_td3_model_q_mlp_1_dense_7_kernel_read_readvariableop=savev2_mlp_td3_model_q_mlp_1_dense_7_bias_read_readvariableopAsavev2_mlp_td3_model_q_1_mlp_2_dense_8_kernel_read_readvariableop?savev2_mlp_td3_model_q_1_mlp_2_dense_8_bias_read_readvariableopAsavev2_mlp_td3_model_q_1_mlp_2_dense_9_kernel_read_readvariableop?savev2_mlp_td3_model_q_1_mlp_2_dense_9_bias_read_readvariableopBsavev2_mlp_td3_model_q_1_mlp_2_dense_10_kernel_read_readvariableop@savev2_mlp_td3_model_q_1_mlp_2_dense_10_bias_read_readvariableopBsavev2_mlp_td3_model_q_1_mlp_2_dense_11_kernel_read_readvariableop@savev2_mlp_td3_model_q_1_mlp_2_dense_11_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *'
dtypes
2
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
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

identity_1Identity_1:output:0*φ
_input_shapesδ
α: :	?::
::
::	Z:Z:
::
::
::	::
::
::
::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	Z: 

_output_shapes
:Z:&	"
 
_output_shapes
:
:!


_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::

_output_shapes
: 
*
₯
B__inference_q_1_layer_call_and_return_conditional_losses_178346408	
inp_0	
inp_1@
,mlp_2_dense_8_matmul_readvariableop_resource:
<
-mlp_2_dense_8_biasadd_readvariableop_resource:	@
,mlp_2_dense_9_matmul_readvariableop_resource:
<
-mlp_2_dense_9_biasadd_readvariableop_resource:	A
-mlp_2_dense_10_matmul_readvariableop_resource:
=
.mlp_2_dense_10_biasadd_readvariableop_resource:	@
-mlp_2_dense_11_matmul_readvariableop_resource:	<
.mlp_2_dense_11_biasadd_readvariableop_resource:
identity’%mlp_2/dense_10/BiasAdd/ReadVariableOp’$mlp_2/dense_10/MatMul/ReadVariableOp’%mlp_2/dense_11/BiasAdd/ReadVariableOp’$mlp_2/dense_11/MatMul/ReadVariableOp’$mlp_2/dense_8/BiasAdd/ReadVariableOp’#mlp_2/dense_8/MatMul/ReadVariableOp’$mlp_2/dense_9/BiasAdd/ReadVariableOp’#mlp_2/dense_9/MatMul/ReadVariableOpV
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????r
concatConcatV2inp_0inp_1concat/axis:output:0*
N*
T0*(
_output_shapes
:?????????
#mlp_2/dense_8/MatMul/ReadVariableOpReadVariableOp,mlp_2_dense_8_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
mlp_2/dense_8/MatMulMatMulconcat:output:0+mlp_2/dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
$mlp_2/dense_8/BiasAdd/ReadVariableOpReadVariableOp-mlp_2_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0‘
mlp_2/dense_8/BiasAddBiasAddmlp_2/dense_8/MatMul:product:0,mlp_2/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????m
mlp_2/dense_8/SeluSelumlp_2/dense_8/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
#mlp_2/dense_9/MatMul/ReadVariableOpReadVariableOp,mlp_2_dense_9_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0 
mlp_2/dense_9/MatMulMatMul mlp_2/dense_8/Selu:activations:0+mlp_2/dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
$mlp_2/dense_9/BiasAdd/ReadVariableOpReadVariableOp-mlp_2_dense_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0‘
mlp_2/dense_9/BiasAddBiasAddmlp_2/dense_9/MatMul:product:0,mlp_2/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????m
mlp_2/dense_9/SeluSelumlp_2/dense_9/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
$mlp_2/dense_10/MatMul/ReadVariableOpReadVariableOp-mlp_2_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0’
mlp_2/dense_10/MatMulMatMul mlp_2/dense_9/Selu:activations:0,mlp_2/dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
%mlp_2/dense_10/BiasAdd/ReadVariableOpReadVariableOp.mlp_2_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0€
mlp_2/dense_10/BiasAddBiasAddmlp_2/dense_10/MatMul:product:0-mlp_2/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????o
mlp_2/dense_10/SeluSelumlp_2/dense_10/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
$mlp_2/dense_11/MatMul/ReadVariableOpReadVariableOp-mlp_2_dense_11_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0’
mlp_2/dense_11/MatMulMatMul!mlp_2/dense_10/Selu:activations:0,mlp_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
%mlp_2/dense_11/BiasAdd/ReadVariableOpReadVariableOp.mlp_2_dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0£
mlp_2/dense_11/BiasAddBiasAddmlp_2/dense_11/MatMul:product:0-mlp_2/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????n
IdentityIdentitymlp_2/dense_11/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????ώ
NoOpNoOp&^mlp_2/dense_10/BiasAdd/ReadVariableOp%^mlp_2/dense_10/MatMul/ReadVariableOp&^mlp_2/dense_11/BiasAdd/ReadVariableOp%^mlp_2/dense_11/MatMul/ReadVariableOp%^mlp_2/dense_8/BiasAdd/ReadVariableOp$^mlp_2/dense_8/MatMul/ReadVariableOp%^mlp_2/dense_9/BiasAdd/ReadVariableOp$^mlp_2/dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:??????????:?????????Z: : : : : : : : 2N
%mlp_2/dense_10/BiasAdd/ReadVariableOp%mlp_2/dense_10/BiasAdd/ReadVariableOp2L
$mlp_2/dense_10/MatMul/ReadVariableOp$mlp_2/dense_10/MatMul/ReadVariableOp2N
%mlp_2/dense_11/BiasAdd/ReadVariableOp%mlp_2/dense_11/BiasAdd/ReadVariableOp2L
$mlp_2/dense_11/MatMul/ReadVariableOp$mlp_2/dense_11/MatMul/ReadVariableOp2L
$mlp_2/dense_8/BiasAdd/ReadVariableOp$mlp_2/dense_8/BiasAdd/ReadVariableOp2J
#mlp_2/dense_8/MatMul/ReadVariableOp#mlp_2/dense_8/MatMul/ReadVariableOp2L
$mlp_2/dense_9/BiasAdd/ReadVariableOp$mlp_2/dense_9/BiasAdd/ReadVariableOp2J
#mlp_2/dense_9/MatMul/ReadVariableOp#mlp_2/dense_9/MatMul/ReadVariableOp:N J
'
_output_shapes
:??????????

_user_specified_nameinp/0:NJ
'
_output_shapes
:?????????Z

_user_specified_nameinp/1
#
²
D__inference_mlp_1_layer_call_and_return_conditional_losses_178346621
obs:
&dense_4_matmul_readvariableop_resource:
6
'dense_4_biasadd_readvariableop_resource:	:
&dense_5_matmul_readvariableop_resource:
6
'dense_5_biasadd_readvariableop_resource:	:
&dense_6_matmul_readvariableop_resource:
6
'dense_6_biasadd_readvariableop_resource:	9
&dense_7_matmul_readvariableop_resource:	5
'dense_7_biasadd_readvariableop_resource:
identity’dense_4/BiasAdd/ReadVariableOp’dense_4/MatMul/ReadVariableOp’dense_5/BiasAdd/ReadVariableOp’dense_5/MatMul/ReadVariableOp’dense_6/BiasAdd/ReadVariableOp’dense_6/MatMul/ReadVariableOp’dense_7/BiasAdd/ReadVariableOp’dense_7/MatMul/ReadVariableOp
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0w
dense_4/MatMulMatMulobs%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????a
dense_4/SeluSeludense_4/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_5/MatMulMatMuldense_4/Selu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????a
dense_5/SeluSeludense_5/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_6/MatMulMatMuldense_5/Selu:activations:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????a
dense_6/SeluSeludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_7/MatMulMatMuldense_6/Selu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_7/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????Κ
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????: : : : : : : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:M I
(
_output_shapes
:?????????

_user_specified_nameobs
΄


E__inference_policy_layer_call_and_return_conditional_losses_178344382
obs 
mlp_178344364:	?
mlp_178344366:	!
mlp_178344368:

mlp_178344370:	!
mlp_178344372:

mlp_178344374:	 
mlp_178344376:	Z
mlp_178344378:Z
identity’mlp/StatefulPartitionedCallΘ
mlp/StatefulPartitionedCallStatefulPartitionedCallobsmlp_178344364mlp_178344366mlp_178344368mlp_178344370mlp_178344372mlp_178344374mlp_178344376mlp_178344378*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_mlp_layer_call_and_return_conditional_losses_178344363s
IdentityIdentity$mlp/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Zd
NoOpNoOp^mlp/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:??????????: : : : : : : : 2:
mlp/StatefulPartitionedCallmlp/StatefulPartitionedCall:L H
'
_output_shapes
:??????????

_user_specified_nameobs
·

Κ
'__inference_q_1_layer_call_fn_178346352	
inp_0	
inp_1
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	
	unknown_6:
identity’StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinp_0inp_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_q_1_layer_call_and_return_conditional_losses_178344955o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:??????????:?????????Z: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:??????????

_user_specified_nameinp/0:NJ
'
_output_shapes
:?????????Z

_user_specified_nameinp/1
Ι#
§
B__inference_mlp_layer_call_and_return_conditional_losses_178346548
obs7
$dense_matmul_readvariableop_resource:	?4
%dense_biasadd_readvariableop_resource:	:
&dense_1_matmul_readvariableop_resource:
6
'dense_1_biasadd_readvariableop_resource:	:
&dense_2_matmul_readvariableop_resource:
6
'dense_2_biasadd_readvariableop_resource:	9
&dense_3_matmul_readvariableop_resource:	Z5
'dense_3_biasadd_readvariableop_resource:Z
identity’dense/BiasAdd/ReadVariableOp’dense/MatMul/ReadVariableOp’dense_1/BiasAdd/ReadVariableOp’dense_1/MatMul/ReadVariableOp’dense_2/BiasAdd/ReadVariableOp’dense_2/MatMul/ReadVariableOp’dense_3/BiasAdd/ReadVariableOp’dense_3/MatMul/ReadVariableOp
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0s
dense/MatMulMatMulobs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????]

dense/SeluSeludense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_1/MatMulMatMuldense/Selu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????a
dense_1/SeluSeludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_2/MatMulMatMuldense_1/Selu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????a
dense_2/SeluSeludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	Z*
dtype0
dense_3/MatMulMatMuldense_2/Selu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:Z*
dtype0
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Zf
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????Zb
IdentityIdentitydense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????ZΖ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:??????????: : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:L H
'
_output_shapes
:??????????

_user_specified_nameobs
Υ’
Ύ
L__inference_mlp_td3_model_layer_call_and_return_conditional_losses_178345935	
inp_0	
inp_1B
/policy_mlp_dense_matmul_readvariableop_resource:	??
0policy_mlp_dense_biasadd_readvariableop_resource:	E
1policy_mlp_dense_1_matmul_readvariableop_resource:
A
2policy_mlp_dense_1_biasadd_readvariableop_resource:	E
1policy_mlp_dense_2_matmul_readvariableop_resource:
A
2policy_mlp_dense_2_biasadd_readvariableop_resource:	D
1policy_mlp_dense_3_matmul_readvariableop_resource:	Z@
2policy_mlp_dense_3_biasadd_readvariableop_resource:ZB
.q_mlp_1_dense_4_matmul_readvariableop_resource:
>
/q_mlp_1_dense_4_biasadd_readvariableop_resource:	B
.q_mlp_1_dense_5_matmul_readvariableop_resource:
>
/q_mlp_1_dense_5_biasadd_readvariableop_resource:	B
.q_mlp_1_dense_6_matmul_readvariableop_resource:
>
/q_mlp_1_dense_6_biasadd_readvariableop_resource:	A
.q_mlp_1_dense_7_matmul_readvariableop_resource:	=
/q_mlp_1_dense_7_biasadd_readvariableop_resource:D
0q_1_mlp_2_dense_8_matmul_readvariableop_resource:
@
1q_1_mlp_2_dense_8_biasadd_readvariableop_resource:	D
0q_1_mlp_2_dense_9_matmul_readvariableop_resource:
@
1q_1_mlp_2_dense_9_biasadd_readvariableop_resource:	E
1q_1_mlp_2_dense_10_matmul_readvariableop_resource:
A
2q_1_mlp_2_dense_10_biasadd_readvariableop_resource:	D
1q_1_mlp_2_dense_11_matmul_readvariableop_resource:	@
2q_1_mlp_2_dense_11_biasadd_readvariableop_resource:
identity

identity_1

identity_2

identity_3’'policy/mlp/dense/BiasAdd/ReadVariableOp’&policy/mlp/dense/MatMul/ReadVariableOp’)policy/mlp/dense_1/BiasAdd/ReadVariableOp’(policy/mlp/dense_1/MatMul/ReadVariableOp’)policy/mlp/dense_2/BiasAdd/ReadVariableOp’(policy/mlp/dense_2/MatMul/ReadVariableOp’)policy/mlp/dense_3/BiasAdd/ReadVariableOp’(policy/mlp/dense_3/MatMul/ReadVariableOp’&q/mlp_1/dense_4/BiasAdd/ReadVariableOp’(q/mlp_1/dense_4/BiasAdd_1/ReadVariableOp’%q/mlp_1/dense_4/MatMul/ReadVariableOp’'q/mlp_1/dense_4/MatMul_1/ReadVariableOp’&q/mlp_1/dense_5/BiasAdd/ReadVariableOp’(q/mlp_1/dense_5/BiasAdd_1/ReadVariableOp’%q/mlp_1/dense_5/MatMul/ReadVariableOp’'q/mlp_1/dense_5/MatMul_1/ReadVariableOp’&q/mlp_1/dense_6/BiasAdd/ReadVariableOp’(q/mlp_1/dense_6/BiasAdd_1/ReadVariableOp’%q/mlp_1/dense_6/MatMul/ReadVariableOp’'q/mlp_1/dense_6/MatMul_1/ReadVariableOp’&q/mlp_1/dense_7/BiasAdd/ReadVariableOp’(q/mlp_1/dense_7/BiasAdd_1/ReadVariableOp’%q/mlp_1/dense_7/MatMul/ReadVariableOp’'q/mlp_1/dense_7/MatMul_1/ReadVariableOp’)q_1/mlp_2/dense_10/BiasAdd/ReadVariableOp’(q_1/mlp_2/dense_10/MatMul/ReadVariableOp’)q_1/mlp_2/dense_11/BiasAdd/ReadVariableOp’(q_1/mlp_2/dense_11/MatMul/ReadVariableOp’(q_1/mlp_2/dense_8/BiasAdd/ReadVariableOp’'q_1/mlp_2/dense_8/MatMul/ReadVariableOp’(q_1/mlp_2/dense_9/BiasAdd/ReadVariableOp’'q_1/mlp_2/dense_9/MatMul/ReadVariableOp
&policy/mlp/dense/MatMul/ReadVariableOpReadVariableOp/policy_mlp_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0
policy/mlp/dense/MatMulMatMulinp_0.policy/mlp/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
'policy/mlp/dense/BiasAdd/ReadVariableOpReadVariableOp0policy_mlp_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ͺ
policy/mlp/dense/BiasAddBiasAdd!policy/mlp/dense/MatMul:product:0/policy/mlp/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????s
policy/mlp/dense/SeluSelu!policy/mlp/dense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
(policy/mlp/dense_1/MatMul/ReadVariableOpReadVariableOp1policy_mlp_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0­
policy/mlp/dense_1/MatMulMatMul#policy/mlp/dense/Selu:activations:00policy/mlp/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
)policy/mlp/dense_1/BiasAdd/ReadVariableOpReadVariableOp2policy_mlp_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0°
policy/mlp/dense_1/BiasAddBiasAdd#policy/mlp/dense_1/MatMul:product:01policy/mlp/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????w
policy/mlp/dense_1/SeluSelu#policy/mlp/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
(policy/mlp/dense_2/MatMul/ReadVariableOpReadVariableOp1policy_mlp_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0―
policy/mlp/dense_2/MatMulMatMul%policy/mlp/dense_1/Selu:activations:00policy/mlp/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
)policy/mlp/dense_2/BiasAdd/ReadVariableOpReadVariableOp2policy_mlp_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0°
policy/mlp/dense_2/BiasAddBiasAdd#policy/mlp/dense_2/MatMul:product:01policy/mlp/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????w
policy/mlp/dense_2/SeluSelu#policy/mlp/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
(policy/mlp/dense_3/MatMul/ReadVariableOpReadVariableOp1policy_mlp_dense_3_matmul_readvariableop_resource*
_output_shapes
:	Z*
dtype0?
policy/mlp/dense_3/MatMulMatMul%policy/mlp/dense_2/Selu:activations:00policy/mlp/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z
)policy/mlp/dense_3/BiasAdd/ReadVariableOpReadVariableOp2policy_mlp_dense_3_biasadd_readvariableop_resource*
_output_shapes
:Z*
dtype0―
policy/mlp/dense_3/BiasAddBiasAdd#policy/mlp/dense_3/MatMul:product:01policy/mlp/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z|
policy/mlp/dense_3/SigmoidSigmoid#policy/mlp/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????ZX
q/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????v
q/concatConcatV2inp_0inp_1q/concat/axis:output:0*
N*
T0*(
_output_shapes
:?????????
%q/mlp_1/dense_4/MatMul/ReadVariableOpReadVariableOp.q_mlp_1_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
q/mlp_1/dense_4/MatMulMatMulq/concat:output:0-q/mlp_1/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
&q/mlp_1/dense_4/BiasAdd/ReadVariableOpReadVariableOp/q_mlp_1_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0§
q/mlp_1/dense_4/BiasAddBiasAdd q/mlp_1/dense_4/MatMul:product:0.q/mlp_1/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????q
q/mlp_1/dense_4/SeluSelu q/mlp_1/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
%q/mlp_1/dense_5/MatMul/ReadVariableOpReadVariableOp.q_mlp_1_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0¦
q/mlp_1/dense_5/MatMulMatMul"q/mlp_1/dense_4/Selu:activations:0-q/mlp_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
&q/mlp_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp/q_mlp_1_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0§
q/mlp_1/dense_5/BiasAddBiasAdd q/mlp_1/dense_5/MatMul:product:0.q/mlp_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????q
q/mlp_1/dense_5/SeluSelu q/mlp_1/dense_5/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
%q/mlp_1/dense_6/MatMul/ReadVariableOpReadVariableOp.q_mlp_1_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0¦
q/mlp_1/dense_6/MatMulMatMul"q/mlp_1/dense_5/Selu:activations:0-q/mlp_1/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
&q/mlp_1/dense_6/BiasAdd/ReadVariableOpReadVariableOp/q_mlp_1_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0§
q/mlp_1/dense_6/BiasAddBiasAdd q/mlp_1/dense_6/MatMul:product:0.q/mlp_1/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????q
q/mlp_1/dense_6/SeluSelu q/mlp_1/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
%q/mlp_1/dense_7/MatMul/ReadVariableOpReadVariableOp.q_mlp_1_dense_7_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0₯
q/mlp_1/dense_7/MatMulMatMul"q/mlp_1/dense_6/Selu:activations:0-q/mlp_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
&q/mlp_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp/q_mlp_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¦
q/mlp_1/dense_7/BiasAddBiasAdd q/mlp_1/dense_7/MatMul:product:0.q/mlp_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z
q_1/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????z

q_1/concatConcatV2inp_0inp_1q_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:?????????
'q_1/mlp_2/dense_8/MatMul/ReadVariableOpReadVariableOp0q_1_mlp_2_dense_8_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
q_1/mlp_2/dense_8/MatMulMatMulq_1/concat:output:0/q_1/mlp_2/dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
(q_1/mlp_2/dense_8/BiasAdd/ReadVariableOpReadVariableOp1q_1_mlp_2_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
q_1/mlp_2/dense_8/BiasAddBiasAdd"q_1/mlp_2/dense_8/MatMul:product:00q_1/mlp_2/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????u
q_1/mlp_2/dense_8/SeluSelu"q_1/mlp_2/dense_8/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
'q_1/mlp_2/dense_9/MatMul/ReadVariableOpReadVariableOp0q_1_mlp_2_dense_9_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0¬
q_1/mlp_2/dense_9/MatMulMatMul$q_1/mlp_2/dense_8/Selu:activations:0/q_1/mlp_2/dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
(q_1/mlp_2/dense_9/BiasAdd/ReadVariableOpReadVariableOp1q_1_mlp_2_dense_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
q_1/mlp_2/dense_9/BiasAddBiasAdd"q_1/mlp_2/dense_9/MatMul:product:00q_1/mlp_2/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????u
q_1/mlp_2/dense_9/SeluSelu"q_1/mlp_2/dense_9/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
(q_1/mlp_2/dense_10/MatMul/ReadVariableOpReadVariableOp1q_1_mlp_2_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0?
q_1/mlp_2/dense_10/MatMulMatMul$q_1/mlp_2/dense_9/Selu:activations:00q_1/mlp_2/dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
)q_1/mlp_2/dense_10/BiasAdd/ReadVariableOpReadVariableOp2q_1_mlp_2_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0°
q_1/mlp_2/dense_10/BiasAddBiasAdd#q_1/mlp_2/dense_10/MatMul:product:01q_1/mlp_2/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????w
q_1/mlp_2/dense_10/SeluSelu#q_1/mlp_2/dense_10/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
(q_1/mlp_2/dense_11/MatMul/ReadVariableOpReadVariableOp1q_1_mlp_2_dense_11_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0?
q_1/mlp_2/dense_11/MatMulMatMul%q_1/mlp_2/dense_10/Selu:activations:00q_1/mlp_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
)q_1/mlp_2/dense_11/BiasAdd/ReadVariableOpReadVariableOp2q_1_mlp_2_dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0―
q_1/mlp_2/dense_11/BiasAddBiasAdd#q_1/mlp_2/dense_11/MatMul:product:01q_1/mlp_2/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z
q/concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????

q/concat_1ConcatV2inp_0policy/mlp/dense_3/Sigmoid:y:0q/concat_1/axis:output:0*
N*
T0*(
_output_shapes
:?????????
'q/mlp_1/dense_4/MatMul_1/ReadVariableOpReadVariableOp.q_mlp_1_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
q/mlp_1/dense_4/MatMul_1MatMulq/concat_1:output:0/q/mlp_1/dense_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
(q/mlp_1/dense_4/BiasAdd_1/ReadVariableOpReadVariableOp/q_mlp_1_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
q/mlp_1/dense_4/BiasAdd_1BiasAdd"q/mlp_1/dense_4/MatMul_1:product:00q/mlp_1/dense_4/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????u
q/mlp_1/dense_4/Selu_1Selu"q/mlp_1/dense_4/BiasAdd_1:output:0*
T0*(
_output_shapes
:?????????
'q/mlp_1/dense_5/MatMul_1/ReadVariableOpReadVariableOp.q_mlp_1_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0¬
q/mlp_1/dense_5/MatMul_1MatMul$q/mlp_1/dense_4/Selu_1:activations:0/q/mlp_1/dense_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
(q/mlp_1/dense_5/BiasAdd_1/ReadVariableOpReadVariableOp/q_mlp_1_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
q/mlp_1/dense_5/BiasAdd_1BiasAdd"q/mlp_1/dense_5/MatMul_1:product:00q/mlp_1/dense_5/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????u
q/mlp_1/dense_5/Selu_1Selu"q/mlp_1/dense_5/BiasAdd_1:output:0*
T0*(
_output_shapes
:?????????
'q/mlp_1/dense_6/MatMul_1/ReadVariableOpReadVariableOp.q_mlp_1_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0¬
q/mlp_1/dense_6/MatMul_1MatMul$q/mlp_1/dense_5/Selu_1:activations:0/q/mlp_1/dense_6/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
(q/mlp_1/dense_6/BiasAdd_1/ReadVariableOpReadVariableOp/q_mlp_1_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
q/mlp_1/dense_6/BiasAdd_1BiasAdd"q/mlp_1/dense_6/MatMul_1:product:00q/mlp_1/dense_6/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????u
q/mlp_1/dense_6/Selu_1Selu"q/mlp_1/dense_6/BiasAdd_1:output:0*
T0*(
_output_shapes
:?????????
'q/mlp_1/dense_7/MatMul_1/ReadVariableOpReadVariableOp.q_mlp_1_dense_7_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0«
q/mlp_1/dense_7/MatMul_1MatMul$q/mlp_1/dense_6/Selu_1:activations:0/q/mlp_1/dense_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
(q/mlp_1/dense_7/BiasAdd_1/ReadVariableOpReadVariableOp/q_mlp_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¬
q/mlp_1/dense_7/BiasAdd_1BiasAdd"q/mlp_1/dense_7/MatMul_1:product:00q/mlp_1/dense_7/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m
IdentityIdentitypolicy/mlp/dense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????Zq

Identity_1Identity q/mlp_1/dense_7/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????t

Identity_2Identity#q_1/mlp_2/dense_11/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????s

Identity_3Identity"q/mlp_1/dense_7/BiasAdd_1:output:0^NoOp*
T0*'
_output_shapes
:?????????
NoOpNoOp(^policy/mlp/dense/BiasAdd/ReadVariableOp'^policy/mlp/dense/MatMul/ReadVariableOp*^policy/mlp/dense_1/BiasAdd/ReadVariableOp)^policy/mlp/dense_1/MatMul/ReadVariableOp*^policy/mlp/dense_2/BiasAdd/ReadVariableOp)^policy/mlp/dense_2/MatMul/ReadVariableOp*^policy/mlp/dense_3/BiasAdd/ReadVariableOp)^policy/mlp/dense_3/MatMul/ReadVariableOp'^q/mlp_1/dense_4/BiasAdd/ReadVariableOp)^q/mlp_1/dense_4/BiasAdd_1/ReadVariableOp&^q/mlp_1/dense_4/MatMul/ReadVariableOp(^q/mlp_1/dense_4/MatMul_1/ReadVariableOp'^q/mlp_1/dense_5/BiasAdd/ReadVariableOp)^q/mlp_1/dense_5/BiasAdd_1/ReadVariableOp&^q/mlp_1/dense_5/MatMul/ReadVariableOp(^q/mlp_1/dense_5/MatMul_1/ReadVariableOp'^q/mlp_1/dense_6/BiasAdd/ReadVariableOp)^q/mlp_1/dense_6/BiasAdd_1/ReadVariableOp&^q/mlp_1/dense_6/MatMul/ReadVariableOp(^q/mlp_1/dense_6/MatMul_1/ReadVariableOp'^q/mlp_1/dense_7/BiasAdd/ReadVariableOp)^q/mlp_1/dense_7/BiasAdd_1/ReadVariableOp&^q/mlp_1/dense_7/MatMul/ReadVariableOp(^q/mlp_1/dense_7/MatMul_1/ReadVariableOp*^q_1/mlp_2/dense_10/BiasAdd/ReadVariableOp)^q_1/mlp_2/dense_10/MatMul/ReadVariableOp*^q_1/mlp_2/dense_11/BiasAdd/ReadVariableOp)^q_1/mlp_2/dense_11/MatMul/ReadVariableOp)^q_1/mlp_2/dense_8/BiasAdd/ReadVariableOp(^q_1/mlp_2/dense_8/MatMul/ReadVariableOp)^q_1/mlp_2/dense_9/BiasAdd/ReadVariableOp(^q_1/mlp_2/dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:??????????:?????????Z: : : : : : : : : : : : : : : : : : : : : : : : 2R
'policy/mlp/dense/BiasAdd/ReadVariableOp'policy/mlp/dense/BiasAdd/ReadVariableOp2P
&policy/mlp/dense/MatMul/ReadVariableOp&policy/mlp/dense/MatMul/ReadVariableOp2V
)policy/mlp/dense_1/BiasAdd/ReadVariableOp)policy/mlp/dense_1/BiasAdd/ReadVariableOp2T
(policy/mlp/dense_1/MatMul/ReadVariableOp(policy/mlp/dense_1/MatMul/ReadVariableOp2V
)policy/mlp/dense_2/BiasAdd/ReadVariableOp)policy/mlp/dense_2/BiasAdd/ReadVariableOp2T
(policy/mlp/dense_2/MatMul/ReadVariableOp(policy/mlp/dense_2/MatMul/ReadVariableOp2V
)policy/mlp/dense_3/BiasAdd/ReadVariableOp)policy/mlp/dense_3/BiasAdd/ReadVariableOp2T
(policy/mlp/dense_3/MatMul/ReadVariableOp(policy/mlp/dense_3/MatMul/ReadVariableOp2P
&q/mlp_1/dense_4/BiasAdd/ReadVariableOp&q/mlp_1/dense_4/BiasAdd/ReadVariableOp2T
(q/mlp_1/dense_4/BiasAdd_1/ReadVariableOp(q/mlp_1/dense_4/BiasAdd_1/ReadVariableOp2N
%q/mlp_1/dense_4/MatMul/ReadVariableOp%q/mlp_1/dense_4/MatMul/ReadVariableOp2R
'q/mlp_1/dense_4/MatMul_1/ReadVariableOp'q/mlp_1/dense_4/MatMul_1/ReadVariableOp2P
&q/mlp_1/dense_5/BiasAdd/ReadVariableOp&q/mlp_1/dense_5/BiasAdd/ReadVariableOp2T
(q/mlp_1/dense_5/BiasAdd_1/ReadVariableOp(q/mlp_1/dense_5/BiasAdd_1/ReadVariableOp2N
%q/mlp_1/dense_5/MatMul/ReadVariableOp%q/mlp_1/dense_5/MatMul/ReadVariableOp2R
'q/mlp_1/dense_5/MatMul_1/ReadVariableOp'q/mlp_1/dense_5/MatMul_1/ReadVariableOp2P
&q/mlp_1/dense_6/BiasAdd/ReadVariableOp&q/mlp_1/dense_6/BiasAdd/ReadVariableOp2T
(q/mlp_1/dense_6/BiasAdd_1/ReadVariableOp(q/mlp_1/dense_6/BiasAdd_1/ReadVariableOp2N
%q/mlp_1/dense_6/MatMul/ReadVariableOp%q/mlp_1/dense_6/MatMul/ReadVariableOp2R
'q/mlp_1/dense_6/MatMul_1/ReadVariableOp'q/mlp_1/dense_6/MatMul_1/ReadVariableOp2P
&q/mlp_1/dense_7/BiasAdd/ReadVariableOp&q/mlp_1/dense_7/BiasAdd/ReadVariableOp2T
(q/mlp_1/dense_7/BiasAdd_1/ReadVariableOp(q/mlp_1/dense_7/BiasAdd_1/ReadVariableOp2N
%q/mlp_1/dense_7/MatMul/ReadVariableOp%q/mlp_1/dense_7/MatMul/ReadVariableOp2R
'q/mlp_1/dense_7/MatMul_1/ReadVariableOp'q/mlp_1/dense_7/MatMul_1/ReadVariableOp2V
)q_1/mlp_2/dense_10/BiasAdd/ReadVariableOp)q_1/mlp_2/dense_10/BiasAdd/ReadVariableOp2T
(q_1/mlp_2/dense_10/MatMul/ReadVariableOp(q_1/mlp_2/dense_10/MatMul/ReadVariableOp2V
)q_1/mlp_2/dense_11/BiasAdd/ReadVariableOp)q_1/mlp_2/dense_11/BiasAdd/ReadVariableOp2T
(q_1/mlp_2/dense_11/MatMul/ReadVariableOp(q_1/mlp_2/dense_11/MatMul/ReadVariableOp2T
(q_1/mlp_2/dense_8/BiasAdd/ReadVariableOp(q_1/mlp_2/dense_8/BiasAdd/ReadVariableOp2R
'q_1/mlp_2/dense_8/MatMul/ReadVariableOp'q_1/mlp_2/dense_8/MatMul/ReadVariableOp2T
(q_1/mlp_2/dense_9/BiasAdd/ReadVariableOp(q_1/mlp_2/dense_9/BiasAdd/ReadVariableOp2R
'q_1/mlp_2/dense_9/MatMul/ReadVariableOp'q_1/mlp_2/dense_9/MatMul/ReadVariableOp:N J
'
_output_shapes
:??????????

_user_specified_nameinp/0:NJ
'
_output_shapes
:?????????Z

_user_specified_nameinp/1
ΐ


E__inference_policy_layer_call_and_return_conditional_losses_178344602
input_1 
mlp_178344584:	?
mlp_178344586:	!
mlp_178344588:

mlp_178344590:	!
mlp_178344592:

mlp_178344594:	 
mlp_178344596:	Z
mlp_178344598:Z
identity’mlp/StatefulPartitionedCallΜ
mlp/StatefulPartitionedCallStatefulPartitionedCallinput_1mlp_178344584mlp_178344586mlp_178344588mlp_178344590mlp_178344592mlp_178344594mlp_178344596mlp_178344598*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_mlp_layer_call_and_return_conditional_losses_178344457s
IdentityIdentity$mlp/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Zd
NoOpNoOp^mlp/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:??????????: : : : : : : : 2:
mlp/StatefulPartitionedCallmlp/StatefulPartitionedCall:P L
'
_output_shapes
:??????????
!
_user_specified_name	input_1
΄


E__inference_policy_layer_call_and_return_conditional_losses_178344520
obs 
mlp_178344502:	?
mlp_178344504:	!
mlp_178344506:

mlp_178344508:	!
mlp_178344510:

mlp_178344512:	 
mlp_178344514:	Z
mlp_178344516:Z
identity’mlp/StatefulPartitionedCallΘ
mlp/StatefulPartitionedCallStatefulPartitionedCallobsmlp_178344502mlp_178344504mlp_178344506mlp_178344508mlp_178344510mlp_178344512mlp_178344514mlp_178344516*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_mlp_layer_call_and_return_conditional_losses_178344457s
IdentityIdentity$mlp/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Zd
NoOpNoOp^mlp/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:??????????: : : : : : : : 2:
mlp/StatefulPartitionedCallmlp/StatefulPartitionedCall:L H
'
_output_shapes
:??????????

_user_specified_nameobs
Γ

Ξ
'__inference_q_1_layer_call_fn_178345138
input_1
input_2
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	
	unknown_6:
identity’StatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_q_1_layer_call_and_return_conditional_losses_178345097o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:??????????:?????????Z: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:??????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????Z
!
_user_specified_name	input_2
?!
©
L__inference_mlp_td3_model_layer_call_and_return_conditional_losses_178345447
inp	
inp_1#
policy_178345383:	?
policy_178345385:	$
policy_178345387:

policy_178345389:	$
policy_178345391:

policy_178345393:	#
policy_178345395:	Z
policy_178345397:Z
q_178345400:

q_178345402:	
q_178345404:

q_178345406:	
q_178345408:

q_178345410:	
q_178345412:	
q_178345414:!
q_1_178345417:

q_1_178345419:	!
q_1_178345421:

q_1_178345423:	!
q_1_178345425:

q_1_178345427:	 
q_1_178345429:	
q_1_178345431:
identity

identity_1

identity_2

identity_3’policy/StatefulPartitionedCall’q/StatefulPartitionedCall’q/StatefulPartitionedCall_1’q_1/StatefulPartitionedCallζ
policy/StatefulPartitionedCallStatefulPartitionedCallinppolicy_178345383policy_178345385policy_178345387policy_178345389policy_178345391policy_178345393policy_178345395policy_178345397*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_policy_layer_call_and_return_conditional_losses_178344520Ό
q/StatefulPartitionedCallStatefulPartitionedCallinpinp_1q_178345400q_178345402q_178345404q_178345406q_178345408q_178345410q_178345412q_178345414*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_q_layer_call_and_return_conditional_losses_178344805Π
q_1/StatefulPartitionedCallStatefulPartitionedCallinpinp_1q_1_178345417q_1_178345419q_1_178345421q_1_178345423q_1_178345425q_1_178345427q_1_178345429q_1_178345431*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_q_1_layer_call_and_return_conditional_losses_178345097ΰ
q/StatefulPartitionedCall_1StatefulPartitionedCallinp'policy/StatefulPartitionedCall:output:0q_178345400q_178345402q_178345404q_178345406q_178345408q_178345410q_178345412q_178345414*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_q_layer_call_and_return_conditional_losses_178344805v
IdentityIdentity'policy/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Zs

Identity_1Identity"q/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????u

Identity_2Identity$q_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????u

Identity_3Identity$q/StatefulPartitionedCall_1:output:0^NoOp*
T0*'
_output_shapes
:?????????Ώ
NoOpNoOp^policy/StatefulPartitionedCall^q/StatefulPartitionedCall^q/StatefulPartitionedCall_1^q_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:??????????:?????????Z: : : : : : : : : : : : : : : : : : : : : : : : 2@
policy/StatefulPartitionedCallpolicy/StatefulPartitionedCall26
q/StatefulPartitionedCallq/StatefulPartitionedCall2:
q/StatefulPartitionedCall_1q/StatefulPartitionedCall_12:
q_1/StatefulPartitionedCallq_1/StatefulPartitionedCall:L H
'
_output_shapes
:??????????

_user_specified_nameinp:LH
'
_output_shapes
:?????????Z

_user_specified_nameinp
#
²
D__inference_mlp_1_layer_call_and_return_conditional_losses_178344737
obs:
&dense_4_matmul_readvariableop_resource:
6
'dense_4_biasadd_readvariableop_resource:	:
&dense_5_matmul_readvariableop_resource:
6
'dense_5_biasadd_readvariableop_resource:	:
&dense_6_matmul_readvariableop_resource:
6
'dense_6_biasadd_readvariableop_resource:	9
&dense_7_matmul_readvariableop_resource:	5
'dense_7_biasadd_readvariableop_resource:
identity’dense_4/BiasAdd/ReadVariableOp’dense_4/MatMul/ReadVariableOp’dense_5/BiasAdd/ReadVariableOp’dense_5/MatMul/ReadVariableOp’dense_6/BiasAdd/ReadVariableOp’dense_6/MatMul/ReadVariableOp’dense_7/BiasAdd/ReadVariableOp’dense_7/MatMul/ReadVariableOp
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0w
dense_4/MatMulMatMulobs%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????a
dense_4/SeluSeludense_4/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_5/MatMulMatMuldense_4/Selu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????a
dense_5/SeluSeludense_5/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_6/MatMulMatMuldense_5/Selu:activations:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????a
dense_6/SeluSeludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_7/MatMulMatMuldense_6/Selu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_7/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????Κ
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????: : : : : : : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:M I
(
_output_shapes
:?????????

_user_specified_nameobs
Α	
Ώ
)__inference_mlp_1_layer_call_fn_178346569
obs
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	
	unknown_6:
identity’StatefulPartitionedCall€
StatefulPartitionedCallStatefulPartitionedCallobsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_mlp_1_layer_call_and_return_conditional_losses_178344644o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:M I
(
_output_shapes
:?????????

_user_specified_nameobs
³

Θ
%__inference_q_layer_call_fn_178346240	
inp_0	
inp_1
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	
	unknown_6:
identity’StatefulPartitionedCallͺ
StatefulPartitionedCallStatefulPartitionedCallinp_0inp_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_q_layer_call_and_return_conditional_losses_178344663o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:??????????:?????????Z: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:??????????

_user_specified_nameinp/0:NJ
'
_output_shapes
:?????????Z

_user_specified_nameinp/1
Ι#
§
B__inference_mlp_layer_call_and_return_conditional_losses_178344363
obs7
$dense_matmul_readvariableop_resource:	?4
%dense_biasadd_readvariableop_resource:	:
&dense_1_matmul_readvariableop_resource:
6
'dense_1_biasadd_readvariableop_resource:	:
&dense_2_matmul_readvariableop_resource:
6
'dense_2_biasadd_readvariableop_resource:	9
&dense_3_matmul_readvariableop_resource:	Z5
'dense_3_biasadd_readvariableop_resource:Z
identity’dense/BiasAdd/ReadVariableOp’dense/MatMul/ReadVariableOp’dense_1/BiasAdd/ReadVariableOp’dense_1/MatMul/ReadVariableOp’dense_2/BiasAdd/ReadVariableOp’dense_2/MatMul/ReadVariableOp’dense_3/BiasAdd/ReadVariableOp’dense_3/MatMul/ReadVariableOp
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0s
dense/MatMulMatMulobs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????]

dense/SeluSeludense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_1/MatMulMatMuldense/Selu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????a
dense_1/SeluSeludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_2/MatMulMatMuldense_1/Selu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????a
dense_2/SeluSeludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	Z*
dtype0
dense_3/MatMulMatMuldense_2/Selu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:Z*
dtype0
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Zf
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????Zb
IdentityIdentitydense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????ZΖ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:??????????: : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:L H
'
_output_shapes
:??????????

_user_specified_nameobs
e
α
%__inference__traced_restore_178346937
file_prefixI
6assignvariableop_mlp_td3_model_policy_mlp_dense_kernel:	?E
6assignvariableop_1_mlp_td3_model_policy_mlp_dense_bias:	N
:assignvariableop_2_mlp_td3_model_policy_mlp_dense_1_kernel:
G
8assignvariableop_3_mlp_td3_model_policy_mlp_dense_1_bias:	N
:assignvariableop_4_mlp_td3_model_policy_mlp_dense_2_kernel:
G
8assignvariableop_5_mlp_td3_model_policy_mlp_dense_2_bias:	M
:assignvariableop_6_mlp_td3_model_policy_mlp_dense_3_kernel:	ZF
8assignvariableop_7_mlp_td3_model_policy_mlp_dense_3_bias:ZK
7assignvariableop_8_mlp_td3_model_q_mlp_1_dense_4_kernel:
D
5assignvariableop_9_mlp_td3_model_q_mlp_1_dense_4_bias:	L
8assignvariableop_10_mlp_td3_model_q_mlp_1_dense_5_kernel:
E
6assignvariableop_11_mlp_td3_model_q_mlp_1_dense_5_bias:	L
8assignvariableop_12_mlp_td3_model_q_mlp_1_dense_6_kernel:
E
6assignvariableop_13_mlp_td3_model_q_mlp_1_dense_6_bias:	K
8assignvariableop_14_mlp_td3_model_q_mlp_1_dense_7_kernel:	D
6assignvariableop_15_mlp_td3_model_q_mlp_1_dense_7_bias:N
:assignvariableop_16_mlp_td3_model_q_1_mlp_2_dense_8_kernel:
G
8assignvariableop_17_mlp_td3_model_q_1_mlp_2_dense_8_bias:	N
:assignvariableop_18_mlp_td3_model_q_1_mlp_2_dense_9_kernel:
G
8assignvariableop_19_mlp_td3_model_q_1_mlp_2_dense_9_bias:	O
;assignvariableop_20_mlp_td3_model_q_1_mlp_2_dense_10_kernel:
H
9assignvariableop_21_mlp_td3_model_q_1_mlp_2_dense_10_bias:	N
;assignvariableop_22_mlp_td3_model_q_1_mlp_2_dense_11_kernel:	G
9assignvariableop_23_mlp_td3_model_q_1_mlp_2_dense_11_bias:
identity_25’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_11’AssignVariableOp_12’AssignVariableOp_13’AssignVariableOp_14’AssignVariableOp_15’AssignVariableOp_16’AssignVariableOp_17’AssignVariableOp_18’AssignVariableOp_19’AssignVariableOp_2’AssignVariableOp_20’AssignVariableOp_21’AssignVariableOp_22’AssignVariableOp_23’AssignVariableOp_3’AssignVariableOp_4’AssignVariableOp_5’AssignVariableOp_6’AssignVariableOp_7’AssignVariableOp_8’AssignVariableOp_9Ϋ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueχBτB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH’
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:‘
AssignVariableOpAssignVariableOp6assignvariableop_mlp_td3_model_policy_mlp_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:₯
AssignVariableOp_1AssignVariableOp6assignvariableop_1_mlp_td3_model_policy_mlp_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_2AssignVariableOp:assignvariableop_2_mlp_td3_model_policy_mlp_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_3AssignVariableOp8assignvariableop_3_mlp_td3_model_policy_mlp_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_4AssignVariableOp:assignvariableop_4_mlp_td3_model_policy_mlp_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_5AssignVariableOp8assignvariableop_5_mlp_td3_model_policy_mlp_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_6AssignVariableOp:assignvariableop_6_mlp_td3_model_policy_mlp_dense_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_7AssignVariableOp8assignvariableop_7_mlp_td3_model_policy_mlp_dense_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_8AssignVariableOp7assignvariableop_8_mlp_td3_model_q_mlp_1_dense_4_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:€
AssignVariableOp_9AssignVariableOp5assignvariableop_9_mlp_td3_model_q_mlp_1_dense_4_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_10AssignVariableOp8assignvariableop_10_mlp_td3_model_q_mlp_1_dense_5_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_11AssignVariableOp6assignvariableop_11_mlp_td3_model_q_mlp_1_dense_5_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_12AssignVariableOp8assignvariableop_12_mlp_td3_model_q_mlp_1_dense_6_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_13AssignVariableOp6assignvariableop_13_mlp_td3_model_q_mlp_1_dense_6_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_14AssignVariableOp8assignvariableop_14_mlp_td3_model_q_mlp_1_dense_7_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_15AssignVariableOp6assignvariableop_15_mlp_td3_model_q_mlp_1_dense_7_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_16AssignVariableOp:assignvariableop_16_mlp_td3_model_q_1_mlp_2_dense_8_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_17AssignVariableOp8assignvariableop_17_mlp_td3_model_q_1_mlp_2_dense_8_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_18AssignVariableOp:assignvariableop_18_mlp_td3_model_q_1_mlp_2_dense_9_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_19AssignVariableOp8assignvariableop_19_mlp_td3_model_q_1_mlp_2_dense_9_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_20AssignVariableOp;assignvariableop_20_mlp_td3_model_q_1_mlp_2_dense_10_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:ͺ
AssignVariableOp_21AssignVariableOp9assignvariableop_21_mlp_td3_model_q_1_mlp_2_dense_10_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_22AssignVariableOp;assignvariableop_22_mlp_td3_model_q_1_mlp_2_dense_11_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:ͺ
AssignVariableOp_23AssignVariableOp9assignvariableop_23_mlp_td3_model_q_1_mlp_2_dense_11_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ί
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_25IdentityIdentity_24:output:0^NoOp_1*
T0*
_output_shapes
: Μ
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_25Identity_25:output:0*E
_input_shapes4
2: : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232(
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
Ί	
Ό
'__inference_mlp_layer_call_fn_178346484
obs
unknown:	?
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	Z
	unknown_6:Z
identity’StatefulPartitionedCall’
StatefulPartitionedCallStatefulPartitionedCallobsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_mlp_layer_call_and_return_conditional_losses_178344457o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:??????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
'
_output_shapes
:??????????

_user_specified_nameobs
Γ#
Ί
D__inference_mlp_2_layer_call_and_return_conditional_losses_178344936
obs:
&dense_8_matmul_readvariableop_resource:
6
'dense_8_biasadd_readvariableop_resource:	:
&dense_9_matmul_readvariableop_resource:
6
'dense_9_biasadd_readvariableop_resource:	;
'dense_10_matmul_readvariableop_resource:
7
(dense_10_biasadd_readvariableop_resource:	:
'dense_11_matmul_readvariableop_resource:	6
(dense_11_biasadd_readvariableop_resource:
identity’dense_10/BiasAdd/ReadVariableOp’dense_10/MatMul/ReadVariableOp’dense_11/BiasAdd/ReadVariableOp’dense_11/MatMul/ReadVariableOp’dense_8/BiasAdd/ReadVariableOp’dense_8/MatMul/ReadVariableOp’dense_9/BiasAdd/ReadVariableOp’dense_9/MatMul/ReadVariableOp
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0w
dense_8/MatMulMatMulobs%dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????a
dense_8/SeluSeludense_8/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_9/MatMulMatMuldense_8/Selu:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????a
dense_9/SeluSeludense_9/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_10/MatMulMatMuldense_9/Selu:activations:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????c
dense_10/SeluSeludense_10/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_11/MatMulMatMuldense_10/Selu:activations:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_11/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????Ξ
NoOpNoOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????: : : : : : : : 2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:M I
(
_output_shapes
:?????????

_user_specified_nameobs
Ι
ͺ
$__inference__wrapped_model_178344324
input_1
input_2P
=mlp_td3_model_policy_mlp_dense_matmul_readvariableop_resource:	?M
>mlp_td3_model_policy_mlp_dense_biasadd_readvariableop_resource:	S
?mlp_td3_model_policy_mlp_dense_1_matmul_readvariableop_resource:
O
@mlp_td3_model_policy_mlp_dense_1_biasadd_readvariableop_resource:	S
?mlp_td3_model_policy_mlp_dense_2_matmul_readvariableop_resource:
O
@mlp_td3_model_policy_mlp_dense_2_biasadd_readvariableop_resource:	R
?mlp_td3_model_policy_mlp_dense_3_matmul_readvariableop_resource:	ZN
@mlp_td3_model_policy_mlp_dense_3_biasadd_readvariableop_resource:ZP
<mlp_td3_model_q_mlp_1_dense_4_matmul_readvariableop_resource:
L
=mlp_td3_model_q_mlp_1_dense_4_biasadd_readvariableop_resource:	P
<mlp_td3_model_q_mlp_1_dense_5_matmul_readvariableop_resource:
L
=mlp_td3_model_q_mlp_1_dense_5_biasadd_readvariableop_resource:	P
<mlp_td3_model_q_mlp_1_dense_6_matmul_readvariableop_resource:
L
=mlp_td3_model_q_mlp_1_dense_6_biasadd_readvariableop_resource:	O
<mlp_td3_model_q_mlp_1_dense_7_matmul_readvariableop_resource:	K
=mlp_td3_model_q_mlp_1_dense_7_biasadd_readvariableop_resource:R
>mlp_td3_model_q_1_mlp_2_dense_8_matmul_readvariableop_resource:
N
?mlp_td3_model_q_1_mlp_2_dense_8_biasadd_readvariableop_resource:	R
>mlp_td3_model_q_1_mlp_2_dense_9_matmul_readvariableop_resource:
N
?mlp_td3_model_q_1_mlp_2_dense_9_biasadd_readvariableop_resource:	S
?mlp_td3_model_q_1_mlp_2_dense_10_matmul_readvariableop_resource:
O
@mlp_td3_model_q_1_mlp_2_dense_10_biasadd_readvariableop_resource:	R
?mlp_td3_model_q_1_mlp_2_dense_11_matmul_readvariableop_resource:	N
@mlp_td3_model_q_1_mlp_2_dense_11_biasadd_readvariableop_resource:
identity

identity_1

identity_2

identity_3’5mlp_td3_model/policy/mlp/dense/BiasAdd/ReadVariableOp’4mlp_td3_model/policy/mlp/dense/MatMul/ReadVariableOp’7mlp_td3_model/policy/mlp/dense_1/BiasAdd/ReadVariableOp’6mlp_td3_model/policy/mlp/dense_1/MatMul/ReadVariableOp’7mlp_td3_model/policy/mlp/dense_2/BiasAdd/ReadVariableOp’6mlp_td3_model/policy/mlp/dense_2/MatMul/ReadVariableOp’7mlp_td3_model/policy/mlp/dense_3/BiasAdd/ReadVariableOp’6mlp_td3_model/policy/mlp/dense_3/MatMul/ReadVariableOp’4mlp_td3_model/q/mlp_1/dense_4/BiasAdd/ReadVariableOp’6mlp_td3_model/q/mlp_1/dense_4/BiasAdd_1/ReadVariableOp’3mlp_td3_model/q/mlp_1/dense_4/MatMul/ReadVariableOp’5mlp_td3_model/q/mlp_1/dense_4/MatMul_1/ReadVariableOp’4mlp_td3_model/q/mlp_1/dense_5/BiasAdd/ReadVariableOp’6mlp_td3_model/q/mlp_1/dense_5/BiasAdd_1/ReadVariableOp’3mlp_td3_model/q/mlp_1/dense_5/MatMul/ReadVariableOp’5mlp_td3_model/q/mlp_1/dense_5/MatMul_1/ReadVariableOp’4mlp_td3_model/q/mlp_1/dense_6/BiasAdd/ReadVariableOp’6mlp_td3_model/q/mlp_1/dense_6/BiasAdd_1/ReadVariableOp’3mlp_td3_model/q/mlp_1/dense_6/MatMul/ReadVariableOp’5mlp_td3_model/q/mlp_1/dense_6/MatMul_1/ReadVariableOp’4mlp_td3_model/q/mlp_1/dense_7/BiasAdd/ReadVariableOp’6mlp_td3_model/q/mlp_1/dense_7/BiasAdd_1/ReadVariableOp’3mlp_td3_model/q/mlp_1/dense_7/MatMul/ReadVariableOp’5mlp_td3_model/q/mlp_1/dense_7/MatMul_1/ReadVariableOp’7mlp_td3_model/q_1/mlp_2/dense_10/BiasAdd/ReadVariableOp’6mlp_td3_model/q_1/mlp_2/dense_10/MatMul/ReadVariableOp’7mlp_td3_model/q_1/mlp_2/dense_11/BiasAdd/ReadVariableOp’6mlp_td3_model/q_1/mlp_2/dense_11/MatMul/ReadVariableOp’6mlp_td3_model/q_1/mlp_2/dense_8/BiasAdd/ReadVariableOp’5mlp_td3_model/q_1/mlp_2/dense_8/MatMul/ReadVariableOp’6mlp_td3_model/q_1/mlp_2/dense_9/BiasAdd/ReadVariableOp’5mlp_td3_model/q_1/mlp_2/dense_9/MatMul/ReadVariableOp³
4mlp_td3_model/policy/mlp/dense/MatMul/ReadVariableOpReadVariableOp=mlp_td3_model_policy_mlp_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0©
%mlp_td3_model/policy/mlp/dense/MatMulMatMulinput_1<mlp_td3_model/policy/mlp/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????±
5mlp_td3_model/policy/mlp/dense/BiasAdd/ReadVariableOpReadVariableOp>mlp_td3_model_policy_mlp_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Τ
&mlp_td3_model/policy/mlp/dense/BiasAddBiasAdd/mlp_td3_model/policy/mlp/dense/MatMul:product:0=mlp_td3_model/policy/mlp/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
#mlp_td3_model/policy/mlp/dense/SeluSelu/mlp_td3_model/policy/mlp/dense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????Έ
6mlp_td3_model/policy/mlp/dense_1/MatMul/ReadVariableOpReadVariableOp?mlp_td3_model_policy_mlp_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Χ
'mlp_td3_model/policy/mlp/dense_1/MatMulMatMul1mlp_td3_model/policy/mlp/dense/Selu:activations:0>mlp_td3_model/policy/mlp/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????΅
7mlp_td3_model/policy/mlp/dense_1/BiasAdd/ReadVariableOpReadVariableOp@mlp_td3_model_policy_mlp_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ϊ
(mlp_td3_model/policy/mlp/dense_1/BiasAddBiasAdd1mlp_td3_model/policy/mlp/dense_1/MatMul:product:0?mlp_td3_model/policy/mlp/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
%mlp_td3_model/policy/mlp/dense_1/SeluSelu1mlp_td3_model/policy/mlp/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:?????????Έ
6mlp_td3_model/policy/mlp/dense_2/MatMul/ReadVariableOpReadVariableOp?mlp_td3_model_policy_mlp_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Ω
'mlp_td3_model/policy/mlp/dense_2/MatMulMatMul3mlp_td3_model/policy/mlp/dense_1/Selu:activations:0>mlp_td3_model/policy/mlp/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????΅
7mlp_td3_model/policy/mlp/dense_2/BiasAdd/ReadVariableOpReadVariableOp@mlp_td3_model_policy_mlp_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ϊ
(mlp_td3_model/policy/mlp/dense_2/BiasAddBiasAdd1mlp_td3_model/policy/mlp/dense_2/MatMul:product:0?mlp_td3_model/policy/mlp/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
%mlp_td3_model/policy/mlp/dense_2/SeluSelu1mlp_td3_model/policy/mlp/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:?????????·
6mlp_td3_model/policy/mlp/dense_3/MatMul/ReadVariableOpReadVariableOp?mlp_td3_model_policy_mlp_dense_3_matmul_readvariableop_resource*
_output_shapes
:	Z*
dtype0Ψ
'mlp_td3_model/policy/mlp/dense_3/MatMulMatMul3mlp_td3_model/policy/mlp/dense_2/Selu:activations:0>mlp_td3_model/policy/mlp/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z΄
7mlp_td3_model/policy/mlp/dense_3/BiasAdd/ReadVariableOpReadVariableOp@mlp_td3_model_policy_mlp_dense_3_biasadd_readvariableop_resource*
_output_shapes
:Z*
dtype0Ω
(mlp_td3_model/policy/mlp/dense_3/BiasAddBiasAdd1mlp_td3_model/policy/mlp/dense_3/MatMul:product:0?mlp_td3_model/policy/mlp/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z
(mlp_td3_model/policy/mlp/dense_3/SigmoidSigmoid1mlp_td3_model/policy/mlp/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????Zf
mlp_td3_model/q/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????
mlp_td3_model/q/concatConcatV2input_1input_2$mlp_td3_model/q/concat/axis:output:0*
N*
T0*(
_output_shapes
:?????????²
3mlp_td3_model/q/mlp_1/dense_4/MatMul/ReadVariableOpReadVariableOp<mlp_td3_model_q_mlp_1_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Ώ
$mlp_td3_model/q/mlp_1/dense_4/MatMulMatMulmlp_td3_model/q/concat:output:0;mlp_td3_model/q/mlp_1/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????―
4mlp_td3_model/q/mlp_1/dense_4/BiasAdd/ReadVariableOpReadVariableOp=mlp_td3_model_q_mlp_1_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ρ
%mlp_td3_model/q/mlp_1/dense_4/BiasAddBiasAdd.mlp_td3_model/q/mlp_1/dense_4/MatMul:product:0<mlp_td3_model/q/mlp_1/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
"mlp_td3_model/q/mlp_1/dense_4/SeluSelu.mlp_td3_model/q/mlp_1/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:?????????²
3mlp_td3_model/q/mlp_1/dense_5/MatMul/ReadVariableOpReadVariableOp<mlp_td3_model_q_mlp_1_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Π
$mlp_td3_model/q/mlp_1/dense_5/MatMulMatMul0mlp_td3_model/q/mlp_1/dense_4/Selu:activations:0;mlp_td3_model/q/mlp_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????―
4mlp_td3_model/q/mlp_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp=mlp_td3_model_q_mlp_1_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ρ
%mlp_td3_model/q/mlp_1/dense_5/BiasAddBiasAdd.mlp_td3_model/q/mlp_1/dense_5/MatMul:product:0<mlp_td3_model/q/mlp_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
"mlp_td3_model/q/mlp_1/dense_5/SeluSelu.mlp_td3_model/q/mlp_1/dense_5/BiasAdd:output:0*
T0*(
_output_shapes
:?????????²
3mlp_td3_model/q/mlp_1/dense_6/MatMul/ReadVariableOpReadVariableOp<mlp_td3_model_q_mlp_1_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Π
$mlp_td3_model/q/mlp_1/dense_6/MatMulMatMul0mlp_td3_model/q/mlp_1/dense_5/Selu:activations:0;mlp_td3_model/q/mlp_1/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????―
4mlp_td3_model/q/mlp_1/dense_6/BiasAdd/ReadVariableOpReadVariableOp=mlp_td3_model_q_mlp_1_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ρ
%mlp_td3_model/q/mlp_1/dense_6/BiasAddBiasAdd.mlp_td3_model/q/mlp_1/dense_6/MatMul:product:0<mlp_td3_model/q/mlp_1/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
"mlp_td3_model/q/mlp_1/dense_6/SeluSelu.mlp_td3_model/q/mlp_1/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:?????????±
3mlp_td3_model/q/mlp_1/dense_7/MatMul/ReadVariableOpReadVariableOp<mlp_td3_model_q_mlp_1_dense_7_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0Ο
$mlp_td3_model/q/mlp_1/dense_7/MatMulMatMul0mlp_td3_model/q/mlp_1/dense_6/Selu:activations:0;mlp_td3_model/q/mlp_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
4mlp_td3_model/q/mlp_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp=mlp_td3_model_q_mlp_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Π
%mlp_td3_model/q/mlp_1/dense_7/BiasAddBiasAdd.mlp_td3_model/q/mlp_1/dense_7/MatMul:product:0<mlp_td3_model/q/mlp_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
mlp_td3_model/q_1/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????
mlp_td3_model/q_1/concatConcatV2input_1input_2&mlp_td3_model/q_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:?????????Ά
5mlp_td3_model/q_1/mlp_2/dense_8/MatMul/ReadVariableOpReadVariableOp>mlp_td3_model_q_1_mlp_2_dense_8_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Ε
&mlp_td3_model/q_1/mlp_2/dense_8/MatMulMatMul!mlp_td3_model/q_1/concat:output:0=mlp_td3_model/q_1/mlp_2/dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????³
6mlp_td3_model/q_1/mlp_2/dense_8/BiasAdd/ReadVariableOpReadVariableOp?mlp_td3_model_q_1_mlp_2_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Χ
'mlp_td3_model/q_1/mlp_2/dense_8/BiasAddBiasAdd0mlp_td3_model/q_1/mlp_2/dense_8/MatMul:product:0>mlp_td3_model/q_1/mlp_2/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
$mlp_td3_model/q_1/mlp_2/dense_8/SeluSelu0mlp_td3_model/q_1/mlp_2/dense_8/BiasAdd:output:0*
T0*(
_output_shapes
:?????????Ά
5mlp_td3_model/q_1/mlp_2/dense_9/MatMul/ReadVariableOpReadVariableOp>mlp_td3_model_q_1_mlp_2_dense_9_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Φ
&mlp_td3_model/q_1/mlp_2/dense_9/MatMulMatMul2mlp_td3_model/q_1/mlp_2/dense_8/Selu:activations:0=mlp_td3_model/q_1/mlp_2/dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????³
6mlp_td3_model/q_1/mlp_2/dense_9/BiasAdd/ReadVariableOpReadVariableOp?mlp_td3_model_q_1_mlp_2_dense_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Χ
'mlp_td3_model/q_1/mlp_2/dense_9/BiasAddBiasAdd0mlp_td3_model/q_1/mlp_2/dense_9/MatMul:product:0>mlp_td3_model/q_1/mlp_2/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
$mlp_td3_model/q_1/mlp_2/dense_9/SeluSelu0mlp_td3_model/q_1/mlp_2/dense_9/BiasAdd:output:0*
T0*(
_output_shapes
:?????????Έ
6mlp_td3_model/q_1/mlp_2/dense_10/MatMul/ReadVariableOpReadVariableOp?mlp_td3_model_q_1_mlp_2_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Ψ
'mlp_td3_model/q_1/mlp_2/dense_10/MatMulMatMul2mlp_td3_model/q_1/mlp_2/dense_9/Selu:activations:0>mlp_td3_model/q_1/mlp_2/dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????΅
7mlp_td3_model/q_1/mlp_2/dense_10/BiasAdd/ReadVariableOpReadVariableOp@mlp_td3_model_q_1_mlp_2_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ϊ
(mlp_td3_model/q_1/mlp_2/dense_10/BiasAddBiasAdd1mlp_td3_model/q_1/mlp_2/dense_10/MatMul:product:0?mlp_td3_model/q_1/mlp_2/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
%mlp_td3_model/q_1/mlp_2/dense_10/SeluSelu1mlp_td3_model/q_1/mlp_2/dense_10/BiasAdd:output:0*
T0*(
_output_shapes
:?????????·
6mlp_td3_model/q_1/mlp_2/dense_11/MatMul/ReadVariableOpReadVariableOp?mlp_td3_model_q_1_mlp_2_dense_11_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0Ψ
'mlp_td3_model/q_1/mlp_2/dense_11/MatMulMatMul3mlp_td3_model/q_1/mlp_2/dense_10/Selu:activations:0>mlp_td3_model/q_1/mlp_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????΄
7mlp_td3_model/q_1/mlp_2/dense_11/BiasAdd/ReadVariableOpReadVariableOp@mlp_td3_model_q_1_mlp_2_dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ω
(mlp_td3_model/q_1/mlp_2/dense_11/BiasAddBiasAdd1mlp_td3_model/q_1/mlp_2/dense_11/MatMul:product:0?mlp_td3_model/q_1/mlp_2/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
mlp_td3_model/q/concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????Ώ
mlp_td3_model/q/concat_1ConcatV2input_1,mlp_td3_model/policy/mlp/dense_3/Sigmoid:y:0&mlp_td3_model/q/concat_1/axis:output:0*
N*
T0*(
_output_shapes
:?????????΄
5mlp_td3_model/q/mlp_1/dense_4/MatMul_1/ReadVariableOpReadVariableOp<mlp_td3_model_q_mlp_1_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Ε
&mlp_td3_model/q/mlp_1/dense_4/MatMul_1MatMul!mlp_td3_model/q/concat_1:output:0=mlp_td3_model/q/mlp_1/dense_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????±
6mlp_td3_model/q/mlp_1/dense_4/BiasAdd_1/ReadVariableOpReadVariableOp=mlp_td3_model_q_mlp_1_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Χ
'mlp_td3_model/q/mlp_1/dense_4/BiasAdd_1BiasAdd0mlp_td3_model/q/mlp_1/dense_4/MatMul_1:product:0>mlp_td3_model/q/mlp_1/dense_4/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
$mlp_td3_model/q/mlp_1/dense_4/Selu_1Selu0mlp_td3_model/q/mlp_1/dense_4/BiasAdd_1:output:0*
T0*(
_output_shapes
:?????????΄
5mlp_td3_model/q/mlp_1/dense_5/MatMul_1/ReadVariableOpReadVariableOp<mlp_td3_model_q_mlp_1_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Φ
&mlp_td3_model/q/mlp_1/dense_5/MatMul_1MatMul2mlp_td3_model/q/mlp_1/dense_4/Selu_1:activations:0=mlp_td3_model/q/mlp_1/dense_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????±
6mlp_td3_model/q/mlp_1/dense_5/BiasAdd_1/ReadVariableOpReadVariableOp=mlp_td3_model_q_mlp_1_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Χ
'mlp_td3_model/q/mlp_1/dense_5/BiasAdd_1BiasAdd0mlp_td3_model/q/mlp_1/dense_5/MatMul_1:product:0>mlp_td3_model/q/mlp_1/dense_5/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
$mlp_td3_model/q/mlp_1/dense_5/Selu_1Selu0mlp_td3_model/q/mlp_1/dense_5/BiasAdd_1:output:0*
T0*(
_output_shapes
:?????????΄
5mlp_td3_model/q/mlp_1/dense_6/MatMul_1/ReadVariableOpReadVariableOp<mlp_td3_model_q_mlp_1_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Φ
&mlp_td3_model/q/mlp_1/dense_6/MatMul_1MatMul2mlp_td3_model/q/mlp_1/dense_5/Selu_1:activations:0=mlp_td3_model/q/mlp_1/dense_6/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????±
6mlp_td3_model/q/mlp_1/dense_6/BiasAdd_1/ReadVariableOpReadVariableOp=mlp_td3_model_q_mlp_1_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Χ
'mlp_td3_model/q/mlp_1/dense_6/BiasAdd_1BiasAdd0mlp_td3_model/q/mlp_1/dense_6/MatMul_1:product:0>mlp_td3_model/q/mlp_1/dense_6/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
$mlp_td3_model/q/mlp_1/dense_6/Selu_1Selu0mlp_td3_model/q/mlp_1/dense_6/BiasAdd_1:output:0*
T0*(
_output_shapes
:?????????³
5mlp_td3_model/q/mlp_1/dense_7/MatMul_1/ReadVariableOpReadVariableOp<mlp_td3_model_q_mlp_1_dense_7_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0Υ
&mlp_td3_model/q/mlp_1/dense_7/MatMul_1MatMul2mlp_td3_model/q/mlp_1/dense_6/Selu_1:activations:0=mlp_td3_model/q/mlp_1/dense_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????°
6mlp_td3_model/q/mlp_1/dense_7/BiasAdd_1/ReadVariableOpReadVariableOp=mlp_td3_model_q_mlp_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Φ
'mlp_td3_model/q/mlp_1/dense_7/BiasAdd_1BiasAdd0mlp_td3_model/q/mlp_1/dense_7/MatMul_1:product:0>mlp_td3_model/q/mlp_1/dense_7/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????{
IdentityIdentity,mlp_td3_model/policy/mlp/dense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????Z

Identity_1Identity.mlp_td3_model/q/mlp_1/dense_7/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????

Identity_2Identity1mlp_td3_model/q_1/mlp_2/dense_11/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????

Identity_3Identity0mlp_td3_model/q/mlp_1/dense_7/BiasAdd_1:output:0^NoOp*
T0*'
_output_shapes
:?????????Ξ
NoOpNoOp6^mlp_td3_model/policy/mlp/dense/BiasAdd/ReadVariableOp5^mlp_td3_model/policy/mlp/dense/MatMul/ReadVariableOp8^mlp_td3_model/policy/mlp/dense_1/BiasAdd/ReadVariableOp7^mlp_td3_model/policy/mlp/dense_1/MatMul/ReadVariableOp8^mlp_td3_model/policy/mlp/dense_2/BiasAdd/ReadVariableOp7^mlp_td3_model/policy/mlp/dense_2/MatMul/ReadVariableOp8^mlp_td3_model/policy/mlp/dense_3/BiasAdd/ReadVariableOp7^mlp_td3_model/policy/mlp/dense_3/MatMul/ReadVariableOp5^mlp_td3_model/q/mlp_1/dense_4/BiasAdd/ReadVariableOp7^mlp_td3_model/q/mlp_1/dense_4/BiasAdd_1/ReadVariableOp4^mlp_td3_model/q/mlp_1/dense_4/MatMul/ReadVariableOp6^mlp_td3_model/q/mlp_1/dense_4/MatMul_1/ReadVariableOp5^mlp_td3_model/q/mlp_1/dense_5/BiasAdd/ReadVariableOp7^mlp_td3_model/q/mlp_1/dense_5/BiasAdd_1/ReadVariableOp4^mlp_td3_model/q/mlp_1/dense_5/MatMul/ReadVariableOp6^mlp_td3_model/q/mlp_1/dense_5/MatMul_1/ReadVariableOp5^mlp_td3_model/q/mlp_1/dense_6/BiasAdd/ReadVariableOp7^mlp_td3_model/q/mlp_1/dense_6/BiasAdd_1/ReadVariableOp4^mlp_td3_model/q/mlp_1/dense_6/MatMul/ReadVariableOp6^mlp_td3_model/q/mlp_1/dense_6/MatMul_1/ReadVariableOp5^mlp_td3_model/q/mlp_1/dense_7/BiasAdd/ReadVariableOp7^mlp_td3_model/q/mlp_1/dense_7/BiasAdd_1/ReadVariableOp4^mlp_td3_model/q/mlp_1/dense_7/MatMul/ReadVariableOp6^mlp_td3_model/q/mlp_1/dense_7/MatMul_1/ReadVariableOp8^mlp_td3_model/q_1/mlp_2/dense_10/BiasAdd/ReadVariableOp7^mlp_td3_model/q_1/mlp_2/dense_10/MatMul/ReadVariableOp8^mlp_td3_model/q_1/mlp_2/dense_11/BiasAdd/ReadVariableOp7^mlp_td3_model/q_1/mlp_2/dense_11/MatMul/ReadVariableOp7^mlp_td3_model/q_1/mlp_2/dense_8/BiasAdd/ReadVariableOp6^mlp_td3_model/q_1/mlp_2/dense_8/MatMul/ReadVariableOp7^mlp_td3_model/q_1/mlp_2/dense_9/BiasAdd/ReadVariableOp6^mlp_td3_model/q_1/mlp_2/dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:??????????:?????????Z: : : : : : : : : : : : : : : : : : : : : : : : 2n
5mlp_td3_model/policy/mlp/dense/BiasAdd/ReadVariableOp5mlp_td3_model/policy/mlp/dense/BiasAdd/ReadVariableOp2l
4mlp_td3_model/policy/mlp/dense/MatMul/ReadVariableOp4mlp_td3_model/policy/mlp/dense/MatMul/ReadVariableOp2r
7mlp_td3_model/policy/mlp/dense_1/BiasAdd/ReadVariableOp7mlp_td3_model/policy/mlp/dense_1/BiasAdd/ReadVariableOp2p
6mlp_td3_model/policy/mlp/dense_1/MatMul/ReadVariableOp6mlp_td3_model/policy/mlp/dense_1/MatMul/ReadVariableOp2r
7mlp_td3_model/policy/mlp/dense_2/BiasAdd/ReadVariableOp7mlp_td3_model/policy/mlp/dense_2/BiasAdd/ReadVariableOp2p
6mlp_td3_model/policy/mlp/dense_2/MatMul/ReadVariableOp6mlp_td3_model/policy/mlp/dense_2/MatMul/ReadVariableOp2r
7mlp_td3_model/policy/mlp/dense_3/BiasAdd/ReadVariableOp7mlp_td3_model/policy/mlp/dense_3/BiasAdd/ReadVariableOp2p
6mlp_td3_model/policy/mlp/dense_3/MatMul/ReadVariableOp6mlp_td3_model/policy/mlp/dense_3/MatMul/ReadVariableOp2l
4mlp_td3_model/q/mlp_1/dense_4/BiasAdd/ReadVariableOp4mlp_td3_model/q/mlp_1/dense_4/BiasAdd/ReadVariableOp2p
6mlp_td3_model/q/mlp_1/dense_4/BiasAdd_1/ReadVariableOp6mlp_td3_model/q/mlp_1/dense_4/BiasAdd_1/ReadVariableOp2j
3mlp_td3_model/q/mlp_1/dense_4/MatMul/ReadVariableOp3mlp_td3_model/q/mlp_1/dense_4/MatMul/ReadVariableOp2n
5mlp_td3_model/q/mlp_1/dense_4/MatMul_1/ReadVariableOp5mlp_td3_model/q/mlp_1/dense_4/MatMul_1/ReadVariableOp2l
4mlp_td3_model/q/mlp_1/dense_5/BiasAdd/ReadVariableOp4mlp_td3_model/q/mlp_1/dense_5/BiasAdd/ReadVariableOp2p
6mlp_td3_model/q/mlp_1/dense_5/BiasAdd_1/ReadVariableOp6mlp_td3_model/q/mlp_1/dense_5/BiasAdd_1/ReadVariableOp2j
3mlp_td3_model/q/mlp_1/dense_5/MatMul/ReadVariableOp3mlp_td3_model/q/mlp_1/dense_5/MatMul/ReadVariableOp2n
5mlp_td3_model/q/mlp_1/dense_5/MatMul_1/ReadVariableOp5mlp_td3_model/q/mlp_1/dense_5/MatMul_1/ReadVariableOp2l
4mlp_td3_model/q/mlp_1/dense_6/BiasAdd/ReadVariableOp4mlp_td3_model/q/mlp_1/dense_6/BiasAdd/ReadVariableOp2p
6mlp_td3_model/q/mlp_1/dense_6/BiasAdd_1/ReadVariableOp6mlp_td3_model/q/mlp_1/dense_6/BiasAdd_1/ReadVariableOp2j
3mlp_td3_model/q/mlp_1/dense_6/MatMul/ReadVariableOp3mlp_td3_model/q/mlp_1/dense_6/MatMul/ReadVariableOp2n
5mlp_td3_model/q/mlp_1/dense_6/MatMul_1/ReadVariableOp5mlp_td3_model/q/mlp_1/dense_6/MatMul_1/ReadVariableOp2l
4mlp_td3_model/q/mlp_1/dense_7/BiasAdd/ReadVariableOp4mlp_td3_model/q/mlp_1/dense_7/BiasAdd/ReadVariableOp2p
6mlp_td3_model/q/mlp_1/dense_7/BiasAdd_1/ReadVariableOp6mlp_td3_model/q/mlp_1/dense_7/BiasAdd_1/ReadVariableOp2j
3mlp_td3_model/q/mlp_1/dense_7/MatMul/ReadVariableOp3mlp_td3_model/q/mlp_1/dense_7/MatMul/ReadVariableOp2n
5mlp_td3_model/q/mlp_1/dense_7/MatMul_1/ReadVariableOp5mlp_td3_model/q/mlp_1/dense_7/MatMul_1/ReadVariableOp2r
7mlp_td3_model/q_1/mlp_2/dense_10/BiasAdd/ReadVariableOp7mlp_td3_model/q_1/mlp_2/dense_10/BiasAdd/ReadVariableOp2p
6mlp_td3_model/q_1/mlp_2/dense_10/MatMul/ReadVariableOp6mlp_td3_model/q_1/mlp_2/dense_10/MatMul/ReadVariableOp2r
7mlp_td3_model/q_1/mlp_2/dense_11/BiasAdd/ReadVariableOp7mlp_td3_model/q_1/mlp_2/dense_11/BiasAdd/ReadVariableOp2p
6mlp_td3_model/q_1/mlp_2/dense_11/MatMul/ReadVariableOp6mlp_td3_model/q_1/mlp_2/dense_11/MatMul/ReadVariableOp2p
6mlp_td3_model/q_1/mlp_2/dense_8/BiasAdd/ReadVariableOp6mlp_td3_model/q_1/mlp_2/dense_8/BiasAdd/ReadVariableOp2n
5mlp_td3_model/q_1/mlp_2/dense_8/MatMul/ReadVariableOp5mlp_td3_model/q_1/mlp_2/dense_8/MatMul/ReadVariableOp2p
6mlp_td3_model/q_1/mlp_2/dense_9/BiasAdd/ReadVariableOp6mlp_td3_model/q_1/mlp_2/dense_9/BiasAdd/ReadVariableOp2n
5mlp_td3_model/q_1/mlp_2/dense_9/MatMul/ReadVariableOp5mlp_td3_model/q_1/mlp_2/dense_9/MatMul/ReadVariableOp:P L
'
_output_shapes
:??????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????Z
!
_user_specified_name	input_2
Μ	
Γ
*__inference_policy_layer_call_fn_178344560
input_1
unknown:	?
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	Z
	unknown_6:Z
identity’StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_policy_layer_call_and_return_conditional_losses_178344520o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:??????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:??????????
!
_user_specified_name	input_1
Γ

Ξ
'__inference_q_1_layer_call_fn_178344974
input_1
input_2
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	
	unknown_6:
identity’StatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_q_1_layer_call_and_return_conditional_losses_178344955o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:??????????:?????????Z: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:??????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????Z
!
_user_specified_name	input_2"ΫL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*’
serving_default
;
input_10
serving_default_input_1:0??????????
;
input_20
serving_default_input_2:0?????????Z<
output_10
StatefulPartitionedCall:0?????????Z<
output_20
StatefulPartitionedCall:1?????????<
output_30
StatefulPartitionedCall:2?????????<
output_40
StatefulPartitionedCall:3?????????tensorflow/serving/predict:ο 
κ
pi
q1
q2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature

signatures"
_tf_keras_model
­
pi
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_model
¬
q
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_model
¬
q
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses"
_tf_keras_model
Φ
!0
"1
#2
$3
%4
&5
'6
(7
)8
*9
+10
,11
-12
.13
/14
015
116
217
318
419
520
621
722
823"
trackable_list_wrapper
Φ
!0
"1
#2
$3
%4
&5
'6
(7
)8
*9
+10
,11
-12
.13
/14
015
116
217
318
419
520
621
722
823"
trackable_list_wrapper
 "
trackable_list_wrapper
Κ
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object
2?
1__inference_mlp_td3_model_layer_call_fn_178345316
1__inference_mlp_td3_model_layer_call_fn_178345760
1__inference_mlp_td3_model_layer_call_fn_178345820
1__inference_mlp_td3_model_layer_call_fn_178345564°
§²£
FullArgSpec&
args
jself
jinp

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ξ2λ
L__inference_mlp_td3_model_layer_call_and_return_conditional_losses_178345935
L__inference_mlp_td3_model_layer_call_and_return_conditional_losses_178346050
L__inference_mlp_td3_model_layer_call_and_return_conditional_losses_178345632
L__inference_mlp_td3_model_layer_call_and_return_conditional_losses_178345700°
§²£
FullArgSpec&
args
jself
jinp

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ΨBΥ
$__inference__wrapped_model_178344324input_1input_2"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
,
>serving_default"
signature_map
Ζ
?dense_layers
@	dense_out
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses"
_tf_keras_layer
X
!0
"1
#2
$3
%4
&5
'6
(7"
trackable_list_wrapper
X
!0
"1
#2
$3
%4
&5
'6
(7"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ζ2γ
*__inference_policy_layer_call_fn_178344401
*__inference_policy_layer_call_fn_178346133
*__inference_policy_layer_call_fn_178346154
*__inference_policy_layer_call_fn_178344560°
§²£
FullArgSpec&
args
jself
jobs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
?2Ο
E__inference_policy_layer_call_and_return_conditional_losses_178346186
E__inference_policy_layer_call_and_return_conditional_losses_178346218
E__inference_policy_layer_call_and_return_conditional_losses_178344581
E__inference_policy_layer_call_and_return_conditional_losses_178344602°
§²£
FullArgSpec&
args
jself
jobs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ζ
Ldense_layers
M	dense_out
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses"
_tf_keras_layer
X
)0
*1
+2
,3
-4
.5
/6
07"
trackable_list_wrapper
X
)0
*1
+2
,3
-4
.5
/6
07"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2Ο
%__inference_q_layer_call_fn_178344682
%__inference_q_layer_call_fn_178346240
%__inference_q_layer_call_fn_178346262
%__inference_q_layer_call_fn_178344846°
§²£
FullArgSpec&
args
jself
jinp

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ύ2»
@__inference_q_layer_call_and_return_conditional_losses_178346296
@__inference_q_layer_call_and_return_conditional_losses_178346330
@__inference_q_layer_call_and_return_conditional_losses_178344870
@__inference_q_layer_call_and_return_conditional_losses_178344894°
§²£
FullArgSpec&
args
jself
jinp

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ζ
Ydense_layers
Z	dense_out
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses"
_tf_keras_layer
X
10
21
32
43
54
65
76
87"
trackable_list_wrapper
X
10
21
32
43
54
65
76
87"
trackable_list_wrapper
 "
trackable_list_wrapper
­
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
Ϊ2Χ
'__inference_q_1_layer_call_fn_178344974
'__inference_q_1_layer_call_fn_178346352
'__inference_q_1_layer_call_fn_178346374
'__inference_q_1_layer_call_fn_178345138°
§²£
FullArgSpec&
args
jself
jinp

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ζ2Γ
B__inference_q_1_layer_call_and_return_conditional_losses_178346408
B__inference_q_1_layer_call_and_return_conditional_losses_178346442
B__inference_q_1_layer_call_and_return_conditional_losses_178345162
B__inference_q_1_layer_call_and_return_conditional_losses_178345186°
§²£
FullArgSpec&
args
jself
jinp

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
8:6	?2%mlp_td3_model/policy/mlp/dense/kernel
2:02#mlp_td3_model/policy/mlp/dense/bias
;:9
2'mlp_td3_model/policy/mlp/dense_1/kernel
4:22%mlp_td3_model/policy/mlp/dense_1/bias
;:9
2'mlp_td3_model/policy/mlp/dense_2/kernel
4:22%mlp_td3_model/policy/mlp/dense_2/bias
::8	Z2'mlp_td3_model/policy/mlp/dense_3/kernel
3:1Z2%mlp_td3_model/policy/mlp/dense_3/bias
8:6
2$mlp_td3_model/q/mlp_1/dense_4/kernel
1:/2"mlp_td3_model/q/mlp_1/dense_4/bias
8:6
2$mlp_td3_model/q/mlp_1/dense_5/kernel
1:/2"mlp_td3_model/q/mlp_1/dense_5/bias
8:6
2$mlp_td3_model/q/mlp_1/dense_6/kernel
1:/2"mlp_td3_model/q/mlp_1/dense_6/bias
7:5	2$mlp_td3_model/q/mlp_1/dense_7/kernel
0:.2"mlp_td3_model/q/mlp_1/dense_7/bias
::8
2&mlp_td3_model/q_1/mlp_2/dense_8/kernel
3:12$mlp_td3_model/q_1/mlp_2/dense_8/bias
::8
2&mlp_td3_model/q_1/mlp_2/dense_9/kernel
3:12$mlp_td3_model/q_1/mlp_2/dense_9/bias
;:9
2'mlp_td3_model/q_1/mlp_2/dense_10/kernel
4:22%mlp_td3_model/q_1/mlp_2/dense_10/bias
::8	2'mlp_td3_model/q_1/mlp_2/dense_11/kernel
3:12%mlp_td3_model/q_1/mlp_2/dense_11/bias
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ΥB?
'__inference_signature_wrapper_178346112input_1input_2"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
5
f0
g1
h2"
trackable_list_wrapper
»

'kernel
(bias
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses"
_tf_keras_layer
X
!0
"1
#2
$3
%4
&5
'6
(7"
trackable_list_wrapper
X
!0
"1
#2
$3
%4
&5
'6
(7"
trackable_list_wrapper
 "
trackable_list_wrapper
­
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
2
'__inference_mlp_layer_call_fn_178346463
'__inference_mlp_layer_call_fn_178346484­
€² 
FullArgSpec&
args
jself
jobs

jtraining
varargs
 
varkw
 
defaults 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
»2Έ
B__inference_mlp_layer_call_and_return_conditional_losses_178346516
B__inference_mlp_layer_call_and_return_conditional_losses_178346548­
€² 
FullArgSpec&
args
jself
jobs

jtraining
varargs
 
varkw
 
defaults 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
t0
u1
v2"
trackable_list_wrapper
»

/kernel
0bias
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses"
_tf_keras_layer
X
)0
*1
+2
,3
-4
.5
/6
07"
trackable_list_wrapper
X
)0
*1
+2
,3
-4
.5
/6
07"
trackable_list_wrapper
 "
trackable_list_wrapper
―
}non_trainable_variables

~layers
metrics
 layer_regularization_losses
layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
2
)__inference_mlp_1_layer_call_fn_178346569
)__inference_mlp_1_layer_call_fn_178346590­
€² 
FullArgSpec&
args
jself
jobs

jtraining
varargs
 
varkw
 
defaults 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Ώ2Ό
D__inference_mlp_1_layer_call_and_return_conditional_losses_178346621
D__inference_mlp_1_layer_call_and_return_conditional_losses_178346652­
€² 
FullArgSpec&
args
jself
jobs

jtraining
varargs
 
varkw
 
defaults 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
8
0
1
2"
trackable_list_wrapper
Α

7kernel
8bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
X
10
21
32
43
54
65
76
87"
trackable_list_wrapper
X
10
21
32
43
54
65
76
87"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
2
)__inference_mlp_2_layer_call_fn_178346673
)__inference_mlp_2_layer_call_fn_178346694­
€² 
FullArgSpec&
args
jself
jobs

jtraining
varargs
 
varkw
 
defaults 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Ώ2Ό
D__inference_mlp_2_layer_call_and_return_conditional_losses_178346725
D__inference_mlp_2_layer_call_and_return_conditional_losses_178346756­
€² 
FullArgSpec&
args
jself
jobs

jtraining
varargs
 
varkw
 
defaults 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Α

!kernel
"bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Α

#kernel
$bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Α

%kernel
&bias
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
+‘&call_and_return_all_conditional_losses"
_tf_keras_layer
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
’non_trainable_variables
£layers
€metrics
 ₯layer_regularization_losses
¦layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
<
f0
g1
h2
@3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Α

)kernel
*bias
§	variables
¨trainable_variables
©regularization_losses
ͺ	keras_api
«__call__
+¬&call_and_return_all_conditional_losses"
_tf_keras_layer
Α

+kernel
,bias
­	variables
?trainable_variables
―regularization_losses
°	keras_api
±__call__
+²&call_and_return_all_conditional_losses"
_tf_keras_layer
Α

-kernel
.bias
³	variables
΄trainable_variables
΅regularization_losses
Ά	keras_api
·__call__
+Έ&call_and_return_all_conditional_losses"
_tf_keras_layer
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ήnon_trainable_variables
Ίlayers
»metrics
 Όlayer_regularization_losses
½layer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
<
t0
u1
v2
M3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Α

1kernel
2bias
Ύ	variables
Ώtrainable_variables
ΐregularization_losses
Α	keras_api
Β__call__
+Γ&call_and_return_all_conditional_losses"
_tf_keras_layer
Α

3kernel
4bias
Δ	variables
Εtrainable_variables
Ζregularization_losses
Η	keras_api
Θ__call__
+Ι&call_and_return_all_conditional_losses"
_tf_keras_layer
Α

5kernel
6bias
Κ	variables
Λtrainable_variables
Μregularization_losses
Ν	keras_api
Ξ__call__
+Ο&call_and_return_all_conditional_losses"
_tf_keras_layer
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Πnon_trainable_variables
Ρlayers
?metrics
 Σlayer_regularization_losses
Τlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
?
0
1
2
Z3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
Έ
Υnon_trainable_variables
Φlayers
Χmetrics
 Ψlayer_regularization_losses
Ωlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Ϊnon_trainable_variables
Ϋlayers
άmetrics
 έlayer_regularization_losses
ήlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
ίnon_trainable_variables
ΰlayers
αmetrics
 βlayer_regularization_losses
γlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
+‘&call_and_return_all_conditional_losses
'‘"call_and_return_conditional_losses"
_generic_user_object
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
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
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
δnon_trainable_variables
εlayers
ζmetrics
 ηlayer_regularization_losses
θlayer_metrics
§	variables
¨trainable_variables
©regularization_losses
«__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
ιnon_trainable_variables
κlayers
λmetrics
 μlayer_regularization_losses
νlayer_metrics
­	variables
?trainable_variables
―regularization_losses
±__call__
+²&call_and_return_all_conditional_losses
'²"call_and_return_conditional_losses"
_generic_user_object
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
ξnon_trainable_variables
οlayers
πmetrics
 ρlayer_regularization_losses
ςlayer_metrics
³	variables
΄trainable_variables
΅regularization_losses
·__call__
+Έ&call_and_return_all_conditional_losses
'Έ"call_and_return_conditional_losses"
_generic_user_object
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
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
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
σnon_trainable_variables
τlayers
υmetrics
 φlayer_regularization_losses
χlayer_metrics
Ύ	variables
Ώtrainable_variables
ΐregularization_losses
Β__call__
+Γ&call_and_return_all_conditional_losses
'Γ"call_and_return_conditional_losses"
_generic_user_object
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
ψnon_trainable_variables
ωlayers
ϊmetrics
 ϋlayer_regularization_losses
όlayer_metrics
Δ	variables
Εtrainable_variables
Ζregularization_losses
Θ__call__
+Ι&call_and_return_all_conditional_losses
'Ι"call_and_return_conditional_losses"
_generic_user_object
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
ύnon_trainable_variables
ώlayers
?metrics
 layer_regularization_losses
layer_metrics
Κ	variables
Λtrainable_variables
Μregularization_losses
Ξ__call__
+Ο&call_and_return_all_conditional_losses
'Ο"call_and_return_conditional_losses"
_generic_user_object
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapperδ
$__inference__wrapped_model_178344324»!"#$%&'()*+,-./012345678X’U
N’K
IF
!
input_1??????????
!
input_2?????????Z
ͺ "Δͺΐ
.
output_1"
output_1?????????Z
.
output_2"
output_2?????????
.
output_3"
output_3?????????
.
output_4"
output_4?????????¬
D__inference_mlp_1_layer_call_and_return_conditional_losses_178346621d)*+,-./01’.
'’$

obs?????????
p 
ͺ "%’"

0?????????
 ¬
D__inference_mlp_1_layer_call_and_return_conditional_losses_178346652d)*+,-./01’.
'’$

obs?????????
p
ͺ "%’"

0?????????
 
)__inference_mlp_1_layer_call_fn_178346569W)*+,-./01’.
'’$

obs?????????
p 
ͺ "?????????
)__inference_mlp_1_layer_call_fn_178346590W)*+,-./01’.
'’$

obs?????????
p
ͺ "?????????¬
D__inference_mlp_2_layer_call_and_return_conditional_losses_178346725d123456781’.
'’$

obs?????????
p 
ͺ "%’"

0?????????
 ¬
D__inference_mlp_2_layer_call_and_return_conditional_losses_178346756d123456781’.
'’$

obs?????????
p
ͺ "%’"

0?????????
 
)__inference_mlp_2_layer_call_fn_178346673W123456781’.
'’$

obs?????????
p 
ͺ "?????????
)__inference_mlp_2_layer_call_fn_178346694W123456781’.
'’$

obs?????????
p
ͺ "?????????©
B__inference_mlp_layer_call_and_return_conditional_losses_178346516c!"#$%&'(0’-
&’#

obs??????????
p 
ͺ "%’"

0?????????Z
 ©
B__inference_mlp_layer_call_and_return_conditional_losses_178346548c!"#$%&'(0’-
&’#

obs??????????
p
ͺ "%’"

0?????????Z
 
'__inference_mlp_layer_call_fn_178346463V!"#$%&'(0’-
&’#

obs??????????
p 
ͺ "?????????Z
'__inference_mlp_layer_call_fn_178346484V!"#$%&'(0’-
&’#

obs??????????
p
ͺ "?????????ZΦ
L__inference_mlp_td3_model_layer_call_and_return_conditional_losses_178345632!"#$%&'()*+,-./012345678\’Y
R’O
IF
!
input_1??????????
!
input_2?????????Z
p 
ͺ "’
’|

0/0?????????Z

0/1?????????

0/2?????????

0/3?????????
 Φ
L__inference_mlp_td3_model_layer_call_and_return_conditional_losses_178345700!"#$%&'()*+,-./012345678\’Y
R’O
IF
!
input_1??????????
!
input_2?????????Z
p
ͺ "’
’|

0/0?????????Z

0/1?????????

0/2?????????

0/3?????????
 ?
L__inference_mlp_td3_model_layer_call_and_return_conditional_losses_178345935!"#$%&'()*+,-./012345678X’U
N’K
EB

inp/0??????????

inp/1?????????Z
p 
ͺ "’
’|

0/0?????????Z

0/1?????????

0/2?????????

0/3?????????
 ?
L__inference_mlp_td3_model_layer_call_and_return_conditional_losses_178346050!"#$%&'()*+,-./012345678X’U
N’K
EB

inp/0??????????

inp/1?????????Z
p
ͺ "’
’|

0/0?????????Z

0/1?????????

0/2?????????

0/3?????????
 §
1__inference_mlp_td3_model_layer_call_fn_178345316ρ!"#$%&'()*+,-./012345678\’Y
R’O
IF
!
input_1??????????
!
input_2?????????Z
p 
ͺ "w’t

0?????????Z

1?????????

2?????????

3?????????§
1__inference_mlp_td3_model_layer_call_fn_178345564ρ!"#$%&'()*+,-./012345678\’Y
R’O
IF
!
input_1??????????
!
input_2?????????Z
p
ͺ "w’t

0?????????Z

1?????????

2?????????

3?????????£
1__inference_mlp_td3_model_layer_call_fn_178345760ν!"#$%&'()*+,-./012345678X’U
N’K
EB

inp/0??????????

inp/1?????????Z
p 
ͺ "w’t

0?????????Z

1?????????

2?????????

3?????????£
1__inference_mlp_td3_model_layer_call_fn_178345820ν!"#$%&'()*+,-./012345678X’U
N’K
EB

inp/0??????????

inp/1?????????Z
p
ͺ "w’t

0?????????Z

1?????????

2?????????

3?????????°
E__inference_policy_layer_call_and_return_conditional_losses_178344581g!"#$%&'(4’1
*’'
!
input_1??????????
p 
ͺ "%’"

0?????????Z
 °
E__inference_policy_layer_call_and_return_conditional_losses_178344602g!"#$%&'(4’1
*’'
!
input_1??????????
p
ͺ "%’"

0?????????Z
 ¬
E__inference_policy_layer_call_and_return_conditional_losses_178346186c!"#$%&'(0’-
&’#

obs??????????
p 
ͺ "%’"

0?????????Z
 ¬
E__inference_policy_layer_call_and_return_conditional_losses_178346218c!"#$%&'(0’-
&’#

obs??????????
p
ͺ "%’"

0?????????Z
 
*__inference_policy_layer_call_fn_178344401Z!"#$%&'(4’1
*’'
!
input_1??????????
p 
ͺ "?????????Z
*__inference_policy_layer_call_fn_178344560Z!"#$%&'(4’1
*’'
!
input_1??????????
p
ͺ "?????????Z
*__inference_policy_layer_call_fn_178346133V!"#$%&'(0’-
&’#

obs??????????
p 
ͺ "?????????Z
*__inference_policy_layer_call_fn_178346154V!"#$%&'(0’-
&’#

obs??????????
p
ͺ "?????????ZΦ
B__inference_q_1_layer_call_and_return_conditional_losses_17834516212345678\’Y
R’O
IF
!
input_1??????????
!
input_2?????????Z
p 
ͺ "%’"

0?????????
 Φ
B__inference_q_1_layer_call_and_return_conditional_losses_17834518612345678\’Y
R’O
IF
!
input_1??????????
!
input_2?????????Z
p
ͺ "%’"

0?????????
 ?
B__inference_q_1_layer_call_and_return_conditional_losses_17834640812345678X’U
N’K
EB

inp/0??????????

inp/1?????????Z
p 
ͺ "%’"

0?????????
 ?
B__inference_q_1_layer_call_and_return_conditional_losses_17834644212345678X’U
N’K
EB

inp/0??????????

inp/1?????????Z
p
ͺ "%’"

0?????????
 ?
'__inference_q_1_layer_call_fn_17834497412345678\’Y
R’O
IF
!
input_1??????????
!
input_2?????????Z
p 
ͺ "??????????
'__inference_q_1_layer_call_fn_17834513812345678\’Y
R’O
IF
!
input_1??????????
!
input_2?????????Z
p
ͺ "?????????©
'__inference_q_1_layer_call_fn_178346352~12345678X’U
N’K
EB

inp/0??????????

inp/1?????????Z
p 
ͺ "?????????©
'__inference_q_1_layer_call_fn_178346374~12345678X’U
N’K
EB

inp/0??????????

inp/1?????????Z
p
ͺ "?????????Τ
@__inference_q_layer_call_and_return_conditional_losses_178344870)*+,-./0\’Y
R’O
IF
!
input_1??????????
!
input_2?????????Z
p 
ͺ "%’"

0?????????
 Τ
@__inference_q_layer_call_and_return_conditional_losses_178344894)*+,-./0\’Y
R’O
IF
!
input_1??????????
!
input_2?????????Z
p
ͺ "%’"

0?????????
 Π
@__inference_q_layer_call_and_return_conditional_losses_178346296)*+,-./0X’U
N’K
EB

inp/0??????????

inp/1?????????Z
p 
ͺ "%’"

0?????????
 Π
@__inference_q_layer_call_and_return_conditional_losses_178346330)*+,-./0X’U
N’K
EB

inp/0??????????

inp/1?????????Z
p
ͺ "%’"

0?????????
 ¬
%__inference_q_layer_call_fn_178344682)*+,-./0\’Y
R’O
IF
!
input_1??????????
!
input_2?????????Z
p 
ͺ "?????????¬
%__inference_q_layer_call_fn_178344846)*+,-./0\’Y
R’O
IF
!
input_1??????????
!
input_2?????????Z
p
ͺ "?????????§
%__inference_q_layer_call_fn_178346240~)*+,-./0X’U
N’K
EB

inp/0??????????

inp/1?????????Z
p 
ͺ "?????????§
%__inference_q_layer_call_fn_178346262~)*+,-./0X’U
N’K
EB

inp/0??????????

inp/1?????????Z
p
ͺ "?????????ψ
'__inference_signature_wrapper_178346112Μ!"#$%&'()*+,-./012345678i’f
’ 
_ͺ\
,
input_1!
input_1??????????
,
input_2!
input_2?????????Z"Δͺΐ
.
output_1"
output_1?????????Z
.
output_2"
output_2?????????
.
output_3"
output_3?????????
.
output_4"
output_4?????????